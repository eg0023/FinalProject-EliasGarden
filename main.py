import numpy as np
import matplotlib.pyplot as plt
import random


class RandomWalk:

    def __init__(self, size):
        self.num_steps = size
        self.obs_points = []
        self.obstacles = []
        self.edges = [[]]
        self.walk = self.rand_avoid(self.num_steps)

    def rand_avoid(self, size):
        """
        makes a random walk that avoids itself, objects and edges

        :param size: number of steps to be taken
        :type size: int
        :return: random walk that avoids itself, objects, and edges
        :rtype: list
        """
        # Creates empty (zeros) array of appropriate size
        steps = np.zeros((2, size))
        # Determines where the edges are by finding the max distance from origin
        self.edges = np.array(self.border(size))
        # Generates random obstacles before walk is determined
        self.generate_obstacles(self.edges, 3)
        # Finds random walk
        steps = self.avoid_walk(steps)
        return steps

    def avoid_walk(self, steps, i=1):
        """
        Recursive function that takes a np.zeros((3, n)) and finds walk that avoids objects, edges, and itself

        :param steps: x and y values set to zero for appropriate len
        :type steps: list
        :param i: current step
        :type i: int
        :return: x and y values that the walk has travelled to
        :rtype: list
        """

        # Checks if all steps have been done
        if i == len(steps[0]):
            return steps
        else:
            # Gets the possible steps that can be taken (pre-randomized)
            poss_steps = self.possible_steps(steps, i)
            # If there are no possible steps then this path is a dead end: return False
            if poss_steps.size == 0:
                return False
            # Iterate through possible steps and see if any will can be finished all steps
            for j in poss_steps:
                # Creates temporary variable
                temp_steps = np.copy(steps)
                temp_steps[:, i] = j
                # Sets the next step to current so that actually 'moves' to next step in next call
                if i < len(steps[0]) - 1:
                    temp_steps[:, i + 1] = temp_steps[:, i]
                # Creates variable w so we only call avoid_walk() once instead of twice
                w = self.avoid_walk(temp_steps, i + 1)
                # If w is a boolean that means that it is a dead end
                # so if it is not a dead end then we can return that call to avoid_walk()
                if not isinstance(w, bool):
                    return w
            # If none of possible steps can be added (all dead ends) then this is essentially a dead end
            return False

    def possible_steps(self, steps, i):
        """
        Finds the possible steps that the walk could take on its next step

        :param steps: current position of all steps taken
        :type steps: list
        :param i: current step
        :type i: int
        :return: possible points that avoid_walk() can move to
        :rtype: list
        """
        # Changes steps from list of x and y values to [x, y] points
        steps = steps.T.tolist()
        # This represents each of the directions able to move to (+/-x, +/-y)
        options = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        possible_choices = []
        # Fetches the points taken up by obstacles
        edge = self.edges
        edge = edge.T.tolist()
        obs = self.obs_points
        # Checks which options are allowed (doesn't hit itself, edges, or obstacles)
        for j in options:
            temp = np.add(steps[i], j)
            temp = temp.tolist()
            if temp not in steps:
                if temp not in edge:
                    if temp not in obs:
                        # If this step is legal move then add to possible_choices
                        possible_choices.append(temp)
        # Where we randomize the direction we go since avoid_walk() will iterate through non-randomly
        np.random.shuffle(possible_choices)
        possible_choices = np.array(possible_choices)
        return possible_choices

    def generate_obstacles(self, border, n):
        """
        Creates objects (points) that are to be avoided

        :param n: number of objects
        :type n: int
        """
        # Makes sure that there are a positive integer of objects
        if n != 0:
            # Makes a collection of points that are the space taken by the object
            for i in range(n):
                # Has to find place for object that does not block the origin (0,0)
                origin = self.random_obstacle_placement(border)
                x = []
                y = []
                # Makes 4x4 object starting at the origin found earlier
                for j in range(4):
                    for k in range(4):
                        x.append(j + origin[0])
                        y.append(k + origin[1])
                # Appends obstacle object to list of obstacles
                self.obstacles.append([x, y])
            # Iterates through the objects to create a list of points occupied by obstacles
            for i in self.obstacles:
                it = np.array(i).T.tolist()
                for x in it:
                    self.obs_points.append(x)

    def random_obstacle_placement(self, border):
        """
        Recursive function that finds a place to put an obstacle that doesn't block origin (0,0,0)

        :return: a place to put an object that doesn't block the origin (0,0,0)
        :rtype: list
        """
        limit = -border[0][0]
        # Finds a place to put obstacle
        origin = [random.randint(-limit, limit - 4), random.randint(-limit, limit - 4)]
        # Checks to see if placement would block origin
        if (0 >= origin[0] >= -4) and (0 >= origin[1] >= -4):
            # If True then call itself and try again
            return self.random_obstacle_placement(border)
        else:
            # Else return position that was found
            return origin

    @staticmethod
    def border(size):
        """
        Creates a list that contains all the points along the border that the walk happens in

        :param size: number of steps
        :type size: int
        :return: points where the border is
        :rtype: list
        """
        # Determines how long to make each side
        length = round(np.sqrt(size)) * 1.75
        h_length = round(length / 2)
        edge = []

        # Finds each side of the box enclosing the walk

        # top
        x = [i for i in range(-h_length, h_length + 1)]
        side_length = len(x)
        y = [h_length] * side_length
        edge.append(x)
        edge.append(y)

        # right
        x = [h_length] * side_length
        y = [i for i in range(h_length, -h_length - 1, -1)]
        edge[0].extend(x)
        edge[1].extend(y)

        # bottom
        x = [i for i in range(h_length, -h_length - 1, -1)]
        y = [-h_length] * side_length
        edge[0].extend(x)
        edge[1].extend(y)

        # left
        x = [-h_length] * side_length
        y = [i for i in range(-h_length, h_length + 1)]
        edge[0].extend(x)
        edge[1].extend(y)

        return edge

    def show_walk(self):
        """
        shows the walk, border and obstacles. Returns nothing
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(*self.walk)
        ax.plot(*self.border(self.num_steps))

        # Iterates through obstacles to show each one
        for i in self.obstacles:
            rectangle = plt.Rectangle((i[0][0], i[1][0]), 3, 3, fc='tab:brown')
            plt.gca().add_patch(rectangle)

        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    # Seeds can be anything or deleted
    # I set seeds so that I could get consistent results for testing
    # Also, without seeds sometimes the walk can take too long to finish (+20 min sometimes)
    # These seeds are guaranteed to return results that are quickly plotted
    random.seed(2)
    np.random.seed(4206)
    q = RandomWalk(225)
    q.show_walk()
