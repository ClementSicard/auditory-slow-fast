(define (domain egocentric-video)
    (:requirements :strips :typing :negative-preconditions)
    (:types
        object hand recipient - object
        tool - object
    )

    (:predicates
        (broken ?x - object) ; in-hand pre
        (crushed ?x - object)
        (patted ?x - object)
        (shaken ?x - object) ; in-hand pre and post
        (blunt ?x - tool)
        (sharp ?x - tool) ; blunt pre
        (smelled ?x - object)
        (thrown ?x - object)
        (in-hand ?x - object) ; TODO
        (on-surface ?x - object) ; TODO
        (dry ?x - object)
        (wet ?x - object) ; dry pre
    )

    (:action break
        :parameters (?x - object)
        :precondition (
            and
            (in-hand ?x)
            (not (broken ?x))
        )
        :effect (broken ?x)
    )

    (:action crush
        :parameters (?x - object)
        :precondition (
            and
            (on-surface ?x)
            (not (crushed ?x))
        )
        :effect (crushed ?x)
    )

    (:action pat
        :parameters (?x - object)
        :precondition (not (patted ?x))
        :effect (patted ?x)
    )

    (:action shake
        :parameters (?x - object)
        :precondition (
            and
            (not (shaken ?x))
            (in-hand ?x)
        )
        :effect (
            and
            (shaken ?x)
            (in-hand ?x)
        )
    )

    (:action sharpen
        :parameters (?x - tool)
        :precondition (
            and
            (blunt ?x)
            (in-hand ?x)
        )
        :effect (and (sharp ?x) (in-hand ?x))
    )

    (:action smell
        :parameters (?x - object)
        :precondition (not (smelled ?x))
        :effect (smelled ?x)
    )

    (:action throw
        :parameters (?x - object)
        :precondition (
            and
            (not (thrown ?x))
            (in-hand ?x)
        )
        :effect (
            and
            (thrown ?x)
            (not (in-hand ?x))
        )
    )

    (:action water
        :parameters (?x - object)
        :precondition (dry ?x)
        :effect (wet ?x)
    )
)
