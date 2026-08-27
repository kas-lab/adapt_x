from typing import overload
from nav2_msgs.action._navigate_to_pose import NavigateToPose_Goal
from nav2_msgs.action._navigate_to_pose import NavigateToPose_Feedback
from nav2_msgs.action._navigate_through_poses import NavigateThroughPoses_Goal as NavThroughPoses_Goal
from nav2_msgs.action._navigate_through_poses import NavigateThroughPoses_Feedback as NavThroughPoses_Feedback


@overload
def construct_repeat_goal(original_goal: NavThroughPoses_Goal,
                          feedback: NavThroughPoses_Feedback | None) -> NavThroughPoses_Goal:
    if (feedback is None):
        return original_goal
    else:
        poses_remaining = feedback.number_of_poses_remaining
        total_poses = len(original_goal.poses)
        cutoff = total_poses - poses_remaining
        remaining_poses = original_goal.poses[cutoff:]
        original_goal.poses = remaining_poses
        return original_goal


@overload
def construct_repeat_goal(original_goal: NavigateToPose_Goal,
                          feedback: NavigateToPose_Feedback | None) -> NavigateToPose_Goal:
    return original_goal


def construct_repeat_goal(original_goal, feedback):
    return original_goal
