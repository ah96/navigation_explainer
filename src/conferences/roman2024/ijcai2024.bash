rosservice call /rosplan_problem_interface/problem_generation_server
rosservice call /rosplan_planner_interface/planning_server
rosservice call /rosplan_parsing_interface/parse_plan_from_file "plan_path: './src/rosplan_demos/rosplan_demos/common/plan.pddl'"
rosservice call /rosplan_parsing_interface/parse_plan
rosservice call /rosplan_plan_dispatcher/dispatch_plan

roslaunch ./tmp/rosplan_full.launch dispatcher:=online domain_path:=./tmp/ijcai2024_domain.rddl problem_path:=./tmp/ijcai2024_instance.rddl planning_language:=RDDL planner_interface:=online_planner_interface planner_command:='/home/robolab/planning_ws/rosplan_noetic/src/rosplan/rosplan_planning_system/common/bin/prost/run_prost_online.sh DOMAIN PROBLEM "[PROST -s 1 -se [IPPC2014]]"'

roslaunch ./tmp/rosplan_full.launch 

/home/robolab/planning_ws/rosplan_noetic/src/rosplan/rosplan_planning_system/common/bin/prost

$(find rosplan_demos)/common/ijcai2024_instance.rddl
