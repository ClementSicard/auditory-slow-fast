import torch

from src.pddl import Action, Predicate

invertibility_tests = [
    (
        Action(
            name="test1",
            preconditions=[
                Predicate(attribute="a", value=True),
                Predicate(attribute="b", value=False),
            ],
            postconditions=[
                Predicate(attribute="a", value=False),
                Predicate(attribute="d", value=True),
            ],
        ),
        ["a", "b", "c", "d"],
    ),
]

vectorization_tests = [
    (
        Action(
            name="test1",
            preconditions=[
                Predicate(attribute="a", value=True),
                Predicate(attribute="b", value=False),
            ],
            postconditions=[
                Predicate(attribute="a", value=False),
                Predicate(attribute="d", value=True),
            ],
        ),
        (
            torch.tensor([1, -1, 0, 0]),
            torch.tensor(
                [-1, 0, 0, 1],
            ),
        ),
    )
]


def test_invertibility():
    """
    Test invertibility of PDDL vectorization.
    """
    for action, attributes in invertibility_tests:
        pre, post = action.vectorize(attributes)

        assert (
            Predicate.predicates_from_vector(pre, attributes) == action.preconditions
        ), f"Pre: {pre} and {action.preconditions} do not match"

        assert (
            Predicate.predicates_from_vector(post, attributes) == action.postconditions
        ), f"Post: {post} and {action.postconditions} do not match"


def test_vectorization():
    """
    Test PDDL vectorization.
    """

    for test in vectorization_tests:
        action, (exp_pre, exp_post) = test
        pre, post = action.vectorize(["a", "b", "c", "d"])

        assert torch.equal(pre, exp_pre), f"Preconditions vector {pre} and {exp_pre} do not match"

        assert torch.equal(post, exp_post), f"Postconditions vector {post} and {exp_post} do not match"
