(define (problem custom_problem)
    (:domain custom_domain)
    (:objects
        take - item
        put - item
        open - item
        close - item
        fold - item
        pour - item
        rip - item
        insert - item
        move - item
        throw - item
        dry - item
        shake - item
        scoop - item
        squeeze - item
        peel - item
        empty - item
        apply - item
        turn-on - item
        turn-off - item
        mix - item
        press - item
        lift - item
        cut - item
        flip - item
        turn - item
        break - item
        fill - item
        hold - item
        touch - item
        drop - item
        stretch - item
        hang - item
        add - item
        divide - item
        wash - item
        attach - item
        remove - item
        scrape - item
        pat - item
    )
    (:init
        (default-state take)
        (default-state put)
        (default-state open)
        (default-state close)
        (default-state fold)
        (default-state pour)
        (default-state rip)
        (default-state insert)
        (default-state move)
        (default-state throw)
        (default-state dry)
        (default-state shake)
        (default-state scoop)
        (default-state squeeze)
        (default-state peel)
        (default-state empty)
        (default-state apply)
        (default-state turn-on)
        (default-state turn-off)
        (default-state mix)
        (default-state press)
        (default-state lift)
        (default-state cut)
        (default-state flip)
        (default-state turn)
        (default-state break)
        (default-state fill)
        (default-state hold)
        (default-state touch)
        (default-state drop)
        (default-state stretch)
        (default-state hang)
        (default-state add)
        (default-state divide)
        (default-state wash)
        (default-state attach)
        (default-state remove)
        (default-state scrape)
        (default-state pat)
    )
    (:goal (and
        (goal-state take)
        (goal-state put)
        (goal-state open)
        (goal-state close)
        (goal-state fold)
        (goal-state pour)
        (goal-state rip)
        (goal-state insert)
        (goal-state move)
        (goal-state throw)
        (goal-state dry)
        (goal-state shake)
        (goal-state scoop)
        (goal-state squeeze)
        (goal-state peel)
        (goal-state empty)
        (goal-state apply)
        (goal-state turn-on)
        (goal-state turn-off)
        (goal-state mix)
        (goal-state press)
        (goal-state lift)
        (goal-state cut)
        (goal-state flip)
        (goal-state turn)
        (goal-state break)
        (goal-state fill)
        (goal-state hold)
        (goal-state touch)
        (goal-state drop)
        (goal-state stretch)
        (goal-state hang)
        (goal-state add)
        (goal-state divide)
        (goal-state wash)
        (goal-state attach)
        (goal-state remove)
        (goal-state scrape)
        (goal-state pat)
    ))
)
