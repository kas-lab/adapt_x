from nav2_msgs.action._navigate_to_pose import NavigateToPose_Goal
from nav2_msgs.action._navigate_to_pose import NavigateToPose_Feedback
from nav2_msgs.action._navigate_through_poses import NavigateThroughPoses_Goal as NavThroughPoses_Goal
from nav2_msgs.action._navigate_through_poses import NavigateThroughPoses_Feedback as NavThroughPoses_Feedback
from functools import singledispatch


@singledispatch
def construct_repeat_goal(original_goal, logger, feedback):
    logger.info(f"generic {str(type(original_goal))}")
    return original_goal


@construct_repeat_goal.register
def _(original_goal: NavThroughPoses_Goal, logger, feedback: NavThroughPoses_Feedback | None):
    logger.info(f"NavThroughPoses_Goal")
    if feedback is None:
        return original_goal
    poses_remaining = feedback.number_of_poses_remaining
    total_poses = len(original_goal.poses)
    cutoff = total_poses - poses_remaining
    original_goal.poses = original_goal.poses[cutoff:]
    return original_goal


@construct_repeat_goal.register
def _(original_goal: NavigateToPose_Goal, logger, feedback: NavigateToPose_Feedback | None):
    return original_goal

