(define (problem egocentric-hand-actions-template)
    (:domain egocentric-hand-actions)
    (:objects
        a - sth
        b - sth
        c - sth
    )
    (:init)
    ; Impossible dummy goal just to allow parsing.
    (:goal
        (and (close a) (far a))
    )
)
