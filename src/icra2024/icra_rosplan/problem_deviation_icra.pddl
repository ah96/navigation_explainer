(define (problem robot_is_extrovert)
(:domain icsr)
(:objects
    tiago - robot
    chair - obstacle
    amar - human
)
(:init
	(is_extrovert tiago)
	(navigating tiago)
	(at_place chair)
	(is_not_detected amar)
)

(:goal (and 
    (explaining_deviation tiago)
    )
)

)
