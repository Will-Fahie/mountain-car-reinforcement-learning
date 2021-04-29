import gym
import numpy as np
import pickle
import time


def get_discrete_state(state):
    """
    Turns current environment state into a discrete state
    """
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))


def main(epsilon=0.5, q_table=None):
    if q_table is None:
        # initialises q table with q values ranging from -2 to 0
        # action_space = move left, do nothing, move right
        q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
    else:
        with open(q_table, "rb") as f:
            q_table = pickle.load(f)

    for episode in range(EPISODES):

        # if episode is a multiple of SHOW_INTERVAL, the environment will be rendered
        if episode % SHOW_INTERVAL == 0:
            render = True
        elif episode % PRINT_INTERVAL == 0:
            print(f"Episode = {episode}")
            render = False
        else:
            render = False

        # gets starting discrete state of environment
        discrete_state = get_discrete_state(env.reset())
        done = False

        while not done:
            # as epsilon decreases, the chance of a random action decreases
            if np.random.random() > epsilon:
                action = np.nanargmax(q_table[discrete_state])
            else:
                action = np.random.randint(0, env.action_space.n)

            # takes action to create new state
            new_state, reward, done, _ = env.step(action)
            new_discrete_state = get_discrete_state(new_state)

            # renders environment every SHOW_INTERVAL
            if render:
                env.render()

            if not done:
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action, )]

                # the q learning equation (The Bellman Equation)
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT + max_future_q)

                # the index in the q table of the current action with current discrete state is assigned the new q value
                q_table[discrete_state + (action, )] = new_q

            # if the new state (after action) achieves the goal position (the flag on the hill) the smallest penalty is
            # given = 0
            elif new_state[0] >= env.goal_position:
                print(f"Goal reached on episode {episode}")
                q_table[discrete_state + (action, )] = 0

            # updates the current state
            discrete_state = new_discrete_state

        # decreases epsilon (randomness)
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon *= epsilon_decay_value

    save_q_table(q_table)


def save_q_table(q_table):
    with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
        pickle.dump(q_table, f)


if __name__ == "__main__":

    # initialises the environment
    env = gym.make("MountainCar-v0")
    env.reset()

    # constants for q learning
    LEARNING_RATE = 0.1
    DISCOUNT = 0.95  # gamma
    EPISODES = 3000

    # the interval/period for each time the environment is rendered or episode is printed
    SHOW_INTERVAL = 1
    PRINT_INTERVAL = 200

    # initially the state is continuous however a q table has a finite number of indexes so it must be made continuous
    DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)  # observation_space = position, velocity
    discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

    # randomness
    START_EPSILON_DECAYING = 1
    END_EPSILON_DECAYING = EPISODES // 2
    epsilon_decay_value = 0.9998

    # can use previously trained q table
    saved_q_table = "qtable-1619721634.pickle"

    main(epsilon=0, q_table=saved_q_table)

    env.close()
