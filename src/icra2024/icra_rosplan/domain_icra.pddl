(define (domain icsr)

(:requirements :strips :typing :fluents :disjunctive-preconditions :durative-actions :negative-preconditions)

(:types 
	robot
	obstacle
	human
)

(:predicates
	(is_extrovert ?r - robot)
	(is_introvert ?r - robot)
	(deviating?r - robot)
	(failing ?r - robot)
	(navigating ?r - robot)
	(stopped ?r - robot)
	(is_moved ?o - obstacle)
	(at_place ?o - obstacle)
	(is_detected ?h - human)
	(is_not_detected ?h - human)
	(explaining_deviation_introvert ?r - robot)
	(explaining_failure_introvert ?r - robot)	
	(explaining_deviation_extrovert ?r - robot)
	(explaining_failure_extrovert ?r - robot)
)

(:durative-action detect_human
	:parameters (?h - human)
	:duration (= ?duration 1)
	:condition (at start (is_not_detected ?h))
	:effect (and
		(at end (is_detected ?h))
		(at end (not (is_not_detected ?h)))
	)
)

(:durative-action move_obstacle
	:parameters (?o - obstacle)
	:duration (= ?duration 1)
	:condition (and 
		(at start (at_place ?o))
	)
	:effect (and
		(at end (is_moved ?o))
		(at end (not (at_place ?o)))
	)
)

(:durative-action deviating
	:parameters (?o - obstacle, ?r - robot)
	:duration (= ?duration 1)
	:condition (and 
		(at start (is_moved ?o))
		(at start (navigating ?r))
		)
	:effect (and
		(at end (deviating ?r))
		(at end (not (navigating ?r)))
	)
)

(:durative-action fail
	:parameters (?o - obstacle, ?r - robot)
	:duration (= ?duration 1)
	:condition (and 
		(at start (is_moved ?o))
		(at start (navigating ?r))
		)
	:effect (and
		(at end (deviating ?r))
		(at end (not (navigating ?r)))
	)
)

(:durative-action stop
	:parameters (?o - obstacle, ?r - robot)
	:duration (= ?duration 1)
	:condition (and 
		(at start (is_moved ?o))
		(at start (navigating ?r))
		)
	:effect (and
		(at end (failure ?r))
		(at end (not (navigating ?r)))
	)
)

(:durative-action explain_deviation_extrovert
	:parameters (?o - obstacle, ?r - robot, ?h - human)
	:duration (= ?duration 1)
	:condition (and 
		(at start (deviating ?r))
		(at start (is_detected ?h))
		(at start (is_extrovert ?r))
		)
	:effect (at end (explaining_deviation ?r))
)

(:durative-action explain_deviation_introvert)
	:parameters (?o - obstacle, ?r - robot, ?h - human)
	:duration (= ?duration 2)
	:condition (and 
		(at start (deviating ?r))
		(at start (is_detected ?h))
		(at start (is_introvert ?r))
		)
	:effect (at end (explaining_deviation ?r))
)

(:durative-action explain_failure
	:parameters (?o - obstacle, ?r - robot, ?h - human)
	:duration (= ?duration 1)
	:condition (and 
		(at start (failure ?r))
		(at start (is_detected ?h))
		(at start (is_extrovert ?r))
		)
	:effect (at end (explaining_failure ?r))
)

)



