from typing import List, Tuple

import pddlpy
import torch
from pydantic import BaseModel


class Predicate(BaseModel):
    """
    Pydantic model to represent a predicate, with an attribute and a value.
    """

    attribute: str
    value: bool

    def __eq__(self, __value: object) -> bool:
        """
        Overrides the default implementation of the `==` operator.
        Two predicates are equal if they have the same name and value.

        Parameters
        ----------
        `__value` : `object`
            The object to compare to.

        Returns
        -------
        `bool`
            Whether the two objects are equal or not.
        """
        return self.attribute == __value.attribute and self.value == __value.value

    def __hash__(self) -> int:
        """
        Overrides the default implementation of the `__hash__` method.

        Returns
        -------
        `int`
            The hash of the predicate.
        """
        return hash((self.attribute, self.value))

    def __repr__(self) -> str:
        """
        Overrides the default implementation of the `__repr__` method.
        Returns JSON representation of the predicate.

        Returns
        -------
        `str`
            The string representation of the predicate.
        """
        return f"Predicate({self.model_dump_json(indent=2)})"

    def __str__(self) -> str:
        """
        Overrides the default implementation of the `__str__` method.

        Returns
        -------
        `str`
            The string representation of the predicate.
        """
        return ("not-" if not self.value else "") + self.attribute

    @staticmethod
    def predicates_from_vector(
        vector: torch.Tensor,
        attributes: List[str],
        to_str: bool = False,
    ) -> List["Predicate"] | List[str]:
        """
        Create a predicate from a vector.

        Parameters
        ----------
        `vector` : `torch.Tensor`
            The vector.
        `attributes` : `List[str]`
            The list of attributes.
        `to_str` : `bool`
            Whether to return a list of predicates or a list of strings.

        Returns
        -------
        `List[Predicate] | List[str]`
            The list of predicate.
        """
        attributes = sorted(attributes)
        assert vector.shape == (len(attributes),), f"Vector shape is {vector.shape} but should be ({len(attributes)},)"

        # assert vector is only composed of -1, 0 or 1
        assert torch.all(torch.abs(vector) <= 1), f"Vector should only contain -1, 0 or 1 but contains {vector}"

        predicates = []

        for i, attr in enumerate(attributes):
            if vector[i] == 1:
                predicates.append(Predicate(attribute=attr, value=True))
            elif vector[i] == -1:
                predicates.append(Predicate(attribute=attr, value=False))

        lst = sorted(predicates, key=lambda p: p.attribute)

        if to_str:
            lst = [str(p) for p in lst]

        return lst


class Action(BaseModel):
    """
    Pydatntic model to represent an action, with a name, preconditions
    and postconditions
    """

    name: str
    preconditions: List[Predicate]
    postconditions: List[Predicate]

    def __hash__(self) -> int:
        """
        Overrides the default implementation of the `__hash__` method.

        Returns
        -------
        `int`
            The hash of the action.
        """
        return hash(
            (
                self.name,
                frozenset(self.preconditions),
                frozenset(self.postconditions),
            ),
        )

    def __repr__(self) -> str:
        """
        Overrides the default implementation of the `__repr__` method.
        Returns JSON representation of the action.

        Returns
        -------
        `str`
            The string representation of the action.
        """
        return f"Action({self.model_dump_json(indent=2)})"

    def get_all_predicates(self) -> List[Predicate]:
        """
        Get all the predicates of the action.

        Returns
        -------
        `List[Predicate]`
            The list of predicates.
        """
        union = set(self.preconditions).union(self.postconditions)
        return list(union)

    def vectorize(self, all_attributes: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorize the action, given the list of attributes.

        Each index in the final vector correspond to an attribute, and the
        vectorization is done by setting the value at that index to 1 if, for this
        action, there exists a predicate with this attribute that is `True`, -1 if
        there exists a predicate with this attribute which is `False`, and 0 if
        there doesn't exist such a predicate with this attribute.

        Parameters
        ----------
        `predicates` : `List[Predicate]`
            The list of predicates.

        Returns
        -------
        `Tuple[torch.Tensor, torch.Tensor]`
            The vectorized action.


        Example
        -------

        ```
        Action({
            "name": "throw",
            "preconditions": [
                {
                    "attribute": "in-hand",
                    "value": true
                },
                {
                    "attribute": "thrown",
                    "value": false
                }
            ],
            "postconditions": [
                {
                    "attribute": "thrown",
                    "value": true
                },
                {
                    "attribute": "in-hand",
                    "value": false
                }
            ]
        })
        ```
        with attributes

        ```py
        ["in-hand", "sharp", "thrown"]
        ```

        will be vectorized as the pair:

        ```py
        tensor([1., 0., -1.]) # Preconditions vector
        tensor([-1., 0., 1.]) # Postconditions vector
        ```
        """

        # The attributes are sorted alphabetically to ensure that the order of the
        # attributes is the same for all actions
        all_attributes = sorted(all_attributes)

        pre_vector = torch.zeros(len(all_attributes))
        post_vector = torch.zeros(len(all_attributes))

        for p in self.preconditions:
            pre_vector[all_attributes.index(p.attribute)] = 1 if p.value else -1
        for p in self.postconditions:
            post_vector[all_attributes.index(p.attribute)] = 1 if p.value else -1

        return pre_vector, post_vector


def parse_pddl(domain_path: str, problem_path: str) -> Tuple[List[Action], List[str]]:
    """
    Parse the PDDL domain and problem files and return the list of actions and
    attributes.

    Parameters
    ----------
    `domain_path` : `str`
        Path to the domain file.

    `problem_path` : `str`
        Path to the problem file.

    Returns
    -------
    `Tuple[List[Action], List[str]]`
        The list of actions and attributes.
    """

    domprob = pddlpy.DomainProblem(domain_path, problem_path)

    # Operators -> actions
    unparsed_actions = domprob.operators()

    actions = []
    attributes = set()

    for unparsed_action in unparsed_actions:
        pre_conds = []
        pos_conds = []

        # Instantiate unparsed action and get the iterator
        a_iterator = domprob.ground_operator(unparsed_action)
        a = next(a_iterator)

        # Add the attributes to the set
        attributes.update([p[0] for p in a.precondition_pos])
        attributes.update([p[0] for p in a.precondition_neg])
        attributes.update([p[0] for p in a.effect_pos])
        attributes.update([p[0] for p in a.effect_neg])

        # Define the preconditions and postconditions from the predicates
        pre_conds.extend(
            [Predicate(attribute=p[0], value=True) for p in a.precondition_pos],
        )
        pre_conds.extend(
            [Predicate(attribute=p[0], value=False) for p in a.precondition_neg],
        )
        pos_conds.extend(
            [Predicate(attribute=p[0], value=True) for p in a.effect_pos],
        )
        pos_conds.extend(
            [Predicate(attribute=p[0], value=False) for p in a.effect_neg],
        )

        # Add a new action to the list with the preconditions and postconditions
        actions.append(
            Action(
                name=unparsed_action,
                preconditions=pre_conds,
                postconditions=pos_conds,
            ),
        )

    # Sort the attributes alphabetically for consistent vectorization
    attributes = sorted(attributes)

    return actions, attributes
