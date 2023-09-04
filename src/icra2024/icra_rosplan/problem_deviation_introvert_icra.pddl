(define (problem deviation_introvert)
(:domain icra)
(:objects
    tiago - robot
    chair - obstacle
    amar - human
)
(:init
	(is_introvert tiago)
	(navigating tiago)
	(not_deviating tiago)
	(at_place chair)
	(is_not_detected amar)
	(question_not_received tiago)
)

(:goal (and 
    (explaining_deviation_introvert tiago)
    )
)

)
