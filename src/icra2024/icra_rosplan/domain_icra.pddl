(define (domain icra)

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
	(not_deviating?r - robot)
	(failing ?r - robot)
	(not_failing ?r - robot)
	(navigating ?r - robot)
	
	(moving ?r - robot)
	(stopped ?r - robot)
	
	(question_received ?r - robot)
	(question_not_received ?r - robot)

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

(:durative-action deviate
	:parameters (?o - obstacle, ?r - robot)
	:duration (= ?duration 1)
	:condition (and 
		(at start (is_moved ?o))
		(at start (navigating ?r))
		(at start (not_deviating ?r))
		)
	:effect (and
		(at end (deviating ?r))
		(at end (not (navigating ?r)))
		(at start (not (not_deviating ?r)))
	)
)

(:durative-action fail
	:parameters (?o - obstacle, ?r - robot)
	:duration (= ?duration 1)
	:condition (and 
		(at start (is_moved ?o))
		(at start (navigating ?r))
		(at start (not_failing ?r))
		)
	:effect (and
		(at end (failing ?r))
		(at end (not (navigating ?r)))
		(at start (not (not_failing ?r)))
	)
)

(:durative-action stop_because_of_deviation
	:parameters (?o - obstacle, ?r - robot)
	:duration (= ?duration 1)
	:condition (and 
		(at start (is_moved ?o))
		(at start (deviating ?r))
		)
	:effect (and
		(at end (stopped ?r))
		(at end (not (moving ?r)))
	)
)

(:durative-action stop_because_of_failure
	:parameters (?o - obstacle, ?r - robot)
	:duration (= ?duration 1)
	:condition (and 
		(at start (is_moved ?o))
		(at start (failing ?r))
		)
	:effect (and
		(at end (stopped ?r))
		(at end (not (moving ?r)))
	)
)

(:durative-action wait_for_question_with_answer
	:parameters (?o - obstacle, ?r - robot)
	:duration (= ?duration 1)
	:condition (and 
		(at start (stopped ?r))
		(at start (is_introvert ?r))
		)
	:effect (and
		(at end (question_received ?r))
		(at end (not (question_not_received ?r)))
	)
)

(:durative-action wait_for_question_without_answer
	:parameters (?o - obstacle, ?r - robot)
	:duration (= ?duration 1)
	:condition (and 
		(at start (stopped ?r))
		(at start (is_introvert ?r))
		)
	:effect (and
		(at end (question_not_received ?r))
		(at end (not (question_received ?r)))
	)
)

(:durative-action explain_deviation_introvert
	:parameters (?o - obstacle, ?r - robot, ?h - human)
	:duration (= ?duration 2)
	:condition (and 
		(at start (stopped ?r))
		(at start (question_received ?r))
		(at start (deviating ?r))
		(at start (is_detected ?h))
		(at start (is_introvert ?r))
		)
	:effect (at end (explaining_deviation_introvert ?r))
)

(:durative-action explain_failure_introvert
	:parameters (?o - obstacle, ?r - robot, ?h - human)
	:duration (= ?duration 2)
	:condition (and 
		(at start (stopped ?r))
		(at start (question_received ?r))
		(at start (failing ?r))
		(at start (is_detected ?h))
		(at start (is_introvert ?r))
		)
	:effect (at end (explaining_failure_introvert ?r))
)

(:durative-action explain_deviation_extrovert
	:parameters (?o - obstacle, ?r - robot, ?h - human)
	:duration (= ?duration 1)
	:condition (and 
		(at start (deviating ?r))
		(at start (is_detected ?h))
		(at start (is_extrovert ?r))
		)
	:effect (at end (explaining_deviation_extrovert ?r))
)

(:durative-action explain_failure_extrovert
	:parameters (?o - obstacle, ?r - robot, ?h - human)
	:duration (= ?duration 1)
	:condition (and 
		(at start (failing ?r))
		(at start (is_detected ?h))
		(at start (is_extrovert ?r))
		)
	:effect (at end (explaining_failure_extrovert ?r))
)

)



