from rrt_lib.rrt_base import RRTBase
import numpy as np

class RRT(RRTBase):
    def __init__(self, X, Q, x_init, x_goal, max_samples, r, prc=0.01):
        """
        Template RRT planner
        :param X: Search Space
        :param Q: list of lengths of edges added to tree
        :param x_init: tuple, initial location
        :param x_goal: tuple, goal location
        :param max_samples: max number of samples to take
        :param r: resolution of points to sample along edge when checking for collisions
        :param prc: probability of checking whether there is a solution
        """
        super().__init__(X, Q, x_init, x_goal, max_samples, r, prc)

    def rrt_search(self):
        rrt_pts = []
        """
        Create and return a Rapidly-exploring Random Tree, keeps expanding until can connect to goal
        https://en.wikipedia.org/wiki/Rapidly-exploring_random_tree
        :return: list representation of path, dict representing edges of tree in form E[child] = parent
        """
        self.add_vertex(0, self.x_init) # 0 is tree number
        self.add_edge(0, self.x_init, None) # 0 is tree number
        while True:
            for q in self.Q:  # iterate over different edge lengths until solution found or time out
                # for i in range(q[1]):  # iterate over number of edges of given length to add
                x_new, x_nearest = self.new_and_near(0, q)

                if x_new is None:
                    continue
                # print(x_new)
                
                # connect shortest valid edge
                if (self.connect_to_point(0, x_nearest, x_new)):
                    rrt_pts.append(x_new)

                solution = self.check_solution()
                if solution[0]:
                    return np.array(solution[1]), np.array(rrt_pts)
