import warnings
from operator import add
import random


class Environment:
    # TODO - do something about code duplication for the multiple recursive methods
    # TODO - read into Cython for computationally heavy recursion
    # TODO - make max reward optional, if None - find the max reward recursively through state_space

    def __init__(self, state_space, max_reward, action_space, action_vectors, start_position=None):
        self.state_space = state_space
        self.current_state_num = -1
        self.total_num_states = -1

        self.__max_reward = max_reward

        self.action_space = action_space
        self.action_vectors = action_vectors

        if start_position is None:
            start_position = self.get_random_position_comp_heavy()
        self.initial_position = start_position
        self.current_position = self.initial_position

        self.initial_observation = self.build_observation()
        self.current_observation = self.initial_observation

        self.last_reward = None

    '''Sets the environment back to initial state'''
    def reset(self):
        self.current_position = self.initial_position
        self.current_observation = self.initial_observation

    '''Used to progress the env by 1 time-frame.
    Returns: observation, reward, done, info'''
    def step(self, action):
        raise NotImplementedError

    '''Used to execute an action in the game'''
    # at the moment this act() method implies movement of an actor through the state_space
    # todo - generalise this method to not imply movement? - could have 2 methods (act_move, act_place)
    def act(self, action):
        self.__test_action(action)

        self.last_reward = self.get_reward()

    '''Returns a matplotlib plot of the game environment'''
    def render(self):
        raise NotImplementedError

    '''Returns a randomly selected action from the set action_space'''
    def sample_action(self):
        return self.action_space[random.randint(0, len(self.action_space) - 1)]

    '''Returns true if the last reward obtained equals the maximum reward possible, otherwise returns false'''
    def is_done(self):
        if self.last_reward == self.__max_reward:
            return True
        else:
            return False

    '''
    Tests if action is possible, executes action if possible
    Returns True if action successful, False if unsuccessful
    '''
    def __test_action(self, action):
        potential_position = Environment.compute_new_position(self.current_position, self.action_vectors[action])

        action_successful = True

        # Test if action is possible
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', RuntimeWarning)

            observation = self.build_observation(self.state_space, potential_position)

            if len(w) > 0:  # action not possible
                action_successful = False

        # Execute action if possible
        if action_successful:
            self.current_position = potential_position
            self.current_observation = observation
        else:
            # Build observation with current position again because it sets other variables
            # Todo - make a test state of build observation that doesn't effect accessible vars if not possible
            self.build_observation()

        return action_successful

    '''Returns a new position adjusted by action_vector, matching the format of old_position'''
    @staticmethod
    def compute_new_position(old_position, action_vector):
        if len(old_position) != len(action_vector):
            raise ValueError('The number of elements withinin old_position and action_vector must be equal')

        return list(map(add, old_position, action_vector))

    '''Returns a combined space representing all possible states, and possible actions at each state'''
    @staticmethod
    def create_state_action_matrix(state_matrix, action_space):
        # TODO - make this recursive for more than 2d state_action spaces
        state_action_matrix = []

        for i in range(0, len(state_matrix)):
            for j in range(0, len(state_matrix[i])):
                state_action_matrix.append(action_space[:])

        return state_action_matrix

    '''Returns a replication of state_space where each value is 0 except at the position specified'''
    def build_observation(self, state_space=None, position=None):
        # Parameter checks
        if state_space is None:
            state_space = self.state_space

        if position is None:
            position = self.current_position

        self.__position_within_space = False
        self.current_state_num = -1
        self.total_num_states = -1

        observation = self.__build_observation_r(state_space, [], position)

        if not self.__position_within_space:
            self.current_state_num = -1
            warnings.warn('position does not exist as a possible position within state_space', RuntimeWarning)

        return observation

    '''Used to search through a state_space for reward at position specified'''
    def get_reward(self, state_space=None, position=None):
        # Parameter checks
        if state_space is None:
            state_space = self.state_space

        if position is None:
            position = self.current_position

        self.__reward = None

        self.__get_reward_r(state_space, [], position)

        if self.__reward is None:
            raise ValueError('position does not exist as a possible position within state_space')

        return self.__reward

    '''Chooses one random position in state space that has a reward of 0
    Returns 1d list corresponding to the index values of the chosen position within state_space'''
    def get_random_position_comp_heavy(self, state_space=None):
        # Parameter checks
        if state_space is None:
            state_space = self.state_space

        # TODO change total0s to be local
        total0s = self.__get_random_position_comp_heavy_r(state_space, [], 0)

        position = random.randint(1, total0s)

        self.__current0s = 0
        return self.__get_random_position_comp_heavy_r2(state_space, [], position)

        #TODO - 2nd get_position method:
        #  loop through state_space, return a list of (list) indexes (A) that have a reward of '0', choose random number between 0 and A's length (i), return A[i]

    '''Recursive part of build_observation function'''
    def __build_observation_r(self, to_iter, index_list, indexes_to_match):
        try:
            iter(to_iter)
            build_list = []
            for i, val in enumerate(to_iter):
                index_list.append(i)
                build_list.append(self.__build_observation_r(val, index_list, indexes_to_match))
                del index_list[-1]

            return build_list
        except TypeError:
            self.total_num_states += 1
            if not self.__position_within_space:
                self.current_state_num += 1
            if index_list == indexes_to_match:
                self.__position_within_space = True
                return 1
            else:
                return 0

    '''Recursive part of get_reward function'''
    def __get_reward_r(self, to_iter, index_list, indexes_to_match):
        if self.__reward is None:
            try:
                iter(to_iter)
                for i, val in enumerate(to_iter):
                    index_list.append(i)
                    if index_list == indexes_to_match:
                        self.__reward = val
                    else:
                        self.__get_reward_r(val, index_list, indexes_to_match)
                    del index_list[-1]
            except TypeError:
                pass

    '''Recursive part of get_random_position_comp_heavy function'''
    def __get_random_position_comp_heavy_r(self, to_iter, index_list, total0s):
        try:
            iter(to_iter)
            for i, val in enumerate(to_iter):
                index_list.append(i)
                total0s = self.__get_random_position_comp_heavy_r(val, index_list, total0s)
                del index_list[-1]
        except TypeError:
            if to_iter == 0:
                total0s += 1

        return total0s

    '''Recursive part of get_random_position_comp_heavy function'''
    def __get_random_position_comp_heavy_r2(self, to_iter, index_list, number_to_stop):
        try:
            for i, val in enumerate(to_iter):
                index_list.append(i)
                position = self.__get_random_position_comp_heavy_r2(val, index_list, number_to_stop)
                if position is not None:
                    return position
                del index_list[-1]
        except TypeError:
            if to_iter == 0:
                self.__current0s += 1
                if self.__current0s == number_to_stop:
                    return index_list
        return None
