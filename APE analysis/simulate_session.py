import numpy as np
import random
from scipy.stats import truncnorm
from copy import deepcopy
import h5py
from tqdm import tqdm  # progress bar


class Session:
    def __init__(self, T_max=150000, dt=50, baiting_probs=np.array([0.15, 0.32, 0.48, 0.65])):
        """Simulate data of one session of the value-based decision-making task.

        args: T_max -> int: Maximum number of time steps in the session. The actual number might be smaller if all
                            trials of all blocks are completed within fewer steps.
              dt -> float: time step size in miliseconds
              baiting_probs -> 1d array of
        """
        self.T_max = T_max  # maximum number of total steps
        self.T = None  # actual number of steps <= T_max
        self.dt = dt  # time step size [ms]

        self.n_trials = None  # total number of trials
        self.n_trials_completed = None  # total number of trials actually completed
        self.n_trials_blocks = None  # number of trials per block of constant baiting probabilities
        self.n_trials_blocks_completed = None  # number of trials per block actually completed

        self.baiting_probs = baiting_probs
        # Sequences of baiting probabilities on the right and left side for every trial in the session.
        # To be sampled as random blocks of the possible values given in baiting_probs by calling
        # set_baiting_prob_sequences().
        self.baiting_probs_blocks_right = np.array([])  # per block
        self.baiting_probs_blocks_left = np.array([])
        self.baiting_probs_trials_right = np.array([])  # per trial (all blocks concatenated)
        self.baiting_probs_trials_left = np.array([])

        # Initialize matrix of one-hot encoded actions over time.
        # To be generated according to a random policy by calling generate_random_actions().
        self.actions = np.zeros((self.T_max, 3))
        # Boolean arrays indicating for every initiation and choice the mouse makes if it was valid or not.
        self.inits_valid = np.empty(0)
        self.choices_valid = np.empty(0)
        # History on a trial level: One-hot encoding over time of (right choice, left choice, reward available right, reward available left, reward collected).
        self.trial_history = np.empty((0,5))
        self.init_times = np.empty(0)  # time between trial availabilities and trial initiations ("initiation times") in ms
        self.deltasteps_choice_init = np.empty(0)  # steps between initiation and choice for all valid completed trials

        # Initialize vectors of collected and available rewards over time.
        self.rewards_collected = np.zeros(self.T_max)
        self.rewards_collected_right = np.zeros(self.T_max)  # Really need that?
        self.rewards_collected_left = np.zeros(self.T_max)  # Really need that?
        self.rewards_avail_right = np.zeros(self.T_max)
        self.rewards_avail_left = np.zeros(self.T_max)


    def set_baiting_prob_sequences(self):
        """Set the sequence of blocks of constant baiting probabilities on the left and the right side for the session.

        The baiting probabilities on the left and the right side are sampled independently and without replacement from
        the possible values given by self.baiting_probs.
        The combinations of baiting probabilities are sampled such that the higher baiting probability cannot be at the
        same side consecutively even if interspaced by block(s) of equal probabilities.
        The number of trials within each block is determined such that 25 rewarded trials are expected per block on
        average plus a random number between 1 and 10 to make transitions between blocks less predictable."""

        # -------- 1) Sample the sequence of combinations of baiting probabilities to the left and right. --------
        def get_valid(remaining, last_non_equal):
            """Determine the set of valid samples of probability combinations based on the last block of non-equal
            probabilities and the remaining options."""
            if last_non_equal is not None:
                valid = []
                for (i, j) in remaining:
                    if last_non_equal[0] < last_non_equal[1]:  # In the last sample of non-equal probabilities, left had a higher one.
                        if not i < j:  # Do not allow for a subsequent block where left again has a higher probability.
                            valid.append((i, j))
                    else:  # In the last sample of non-equal probabilities, right had a higher one.
                        if not i > j:
                            valid.append((i, j))

            else:  # There has not yet been a combination of non-equal probabilities, thus all remaining ones are valid.
                valid = [(i, j) for (i, j) in remaining]

            return valid

        remaining_combis = set([(i, j) for i in range(4) for j in range(4)])  # All possible combinations (as "matrix indices").
        last_non_eq = None

        while remaining_combis:
            valid = get_valid(remaining_combis, last_non_eq)
            if not valid:
                raise ValueError("No valid sequence of baiting probabilities found. Restart.")
            next_combi = random.choice(valid)
            #prob_combi_sequence.append(next_combi)
            remaining_combis.remove(next_combi)
            # Get the corresponding baiting probabilities.
            self.baiting_probs_blocks_right = np.append(self.baiting_probs_blocks_right, self.baiting_probs[next_combi[0]])
            self.baiting_probs_blocks_left = np.append(self.baiting_probs_blocks_left, self.baiting_probs[next_combi[1]])
            # If a combination of non-equal probabilities has been sampled, update last_non_equal.
            if next_combi[0] != next_combi[1]:
                last_non_eq = next_combi


        # -------- 2) Determine the number of trials in each block of a given baiting probability combination. --------
        n_trials_25 = 25/self.baiting_probs
        n_trials_25 = n_trials_25.astype(int)  # Only integer numbers of trials possible.
        delta_p = np.abs(self.baiting_probs_blocks_right - self.baiting_probs_blocks_left)

        w_low = (1 - delta_p)/2
        w_high = 1 - w_low

        indices_low_side = np.array([np.where(self.baiting_probs == low_prob)[0][0]
                                     for low_prob in np.minimum(self.baiting_probs_blocks_right, self.baiting_probs_blocks_left)])
        indices_high_side = np.array([np.where(self.baiting_probs == high_prob)[0][0]
                                      for high_prob in np.maximum(self.baiting_probs_blocks_right, self.baiting_probs_blocks_left)])

        n_trials_blocks = w_low * n_trials_25[indices_low_side] + w_high * n_trials_25[indices_high_side]

        # If the difference in baiting probabilities between the two sides is at least 0.5, add one more trial in the block.
        n_trials_blocks[np.where(delta_p >= 0.5)[0]] += 1

        # For every block, add a random number of 1-10 trials.
        n_trials_blocks += np.random.randint(1, 11, size=len(n_trials_blocks))
        self.n_trials_blocks = n_trials_blocks
        self.n_trials_blocks = self.n_trials_blocks.astype(int)
        self.n_trials = int(np.sum(self.n_trials_blocks))

        # Set the baiting probabilities for all trials in all blocks.
        self.baiting_probs_trials_right = np.repeat(self.baiting_probs_blocks_right, self.n_trials_blocks)
        self.baiting_probs_trials_left = np.repeat(self.baiting_probs_blocks_left, self.n_trials_blocks)


    def generate_random_actions(self, init_time_mean=2.5, choice_time_mean=500, choice_time_std=np.sqrt(60**2 + 100**2),
                                multi_init_prob=1e-3, multi_choice_prob=1e-3,
                                max_choice_time=10, min_ITI=3., early_init_prob=1e-3):
        """Generate the mouse's actions based on a completely random policy (independent of its action-value-estimates).

        args: init_time_mean -> float: Mean initiation time in seconds. The initiation time is the time between a trial
                                       is made available and the next trial initiation. The actual initiation times are
                                       sampled from an exponential distribution with the given mean.
              choice_time_mean -> float: Mean time window between a valid trial initiation and making a choice
                                         (= reaction + movement time).
              choice_time_std -> float: Standard deviation of the time window between a valid trial initiation and
                                        making a choice.
              The actual choice times are sampled from a normal distribution truncated at dt (one time step) at the
              lower end.
              multi_init_prob -> float: Probability of (unnecessarily) again initiating a new trial after a trial
                                        initiation before a choice has been made (poking into the center port
                                        consecutively) at every step after a first trial initiation. Possibility for
                                        that can be switched of by setting to 0.
              multi_choice_prob -> float: Probability of making another choice to either side at every time step after
                                          already having made a choice in a given trial and before having initiated a
                                          new one. This is an invalid choice and will never be rewarded. Can be turnt
                                          off by setting to 0.
              max_choice_time -> float or None: Maximum time period after a trial initiation within which a choice can
                                                be made in seconds. If no choice is made within that time, the trial
                                                ends as a "miss". Can be turnt off by setting to None.
              min_ITI -> float: Minimum inter-trial-interval in seconds. Can be turnt off by setting to 0.
              early_init_prob -> float: Probability of initiating a trial too early, that is before the minimum ITI has
                                        passed since the most recent trial initiation.

        Sets self.actions -> 2d array of shape (T, 3): 3-dimensional vector of one-hot encoded action taken at every
                                                      time step, where
                                                      (0, 0, 0) -> no action
                                                      (1, 0, 0) -> trial initiation (center poke)
                                                      (0, 1, 0) -> right choice
                                                      (0, 0, 1) -> left choice
        """
        # Initialize action matrix without any actions.
        self.actions = np.zeros((self.T_max, 3))
        self.inits_valid = np.empty(0)
        self.choices_valid = np.empty(0)
        self.init_times = np.empty(0)
        self.sampled_rmtimes = np.empty(0)
        #self.trial_history = np.empty((0, 5))
        # Convert maximum allowed time to make a choice after a trial initiation into number of steps.
        if max_choice_time is not None:
            max_choice_steps = int(max_choice_time * 1e3 / self.dt)

        ITI_steps = int(min_ITI * 1e3 / self.dt)
        prev_init_step = - ITI_steps

        num_trials = 0  # number of trials initiated so far
        in_trial = False
        trial_avail = True
        trial_avail_step = 0
        pass_to_next = False

        # probability of initiating a trial at every time step when a trial is available
        init_prob = 1 - np.exp(-1/(init_time_mean*1e3) * self.dt)

        # ----- Generate the mouses random actions in accordance with the experimental paradigm. -----
        for step in tqdm(range(self.T_max)):
            if num_trials > self.n_trials:  # Stop if all available trials completed before having reached the maximum number of steps.
                num_trials -= 1
                break

            if pass_to_next:
                pass_to_next = False
                continue

            # Check if a new trial is available.
            if step >= prev_init_step + ITI_steps:
                trial_avail = True
                trial_avail_step = prev_init_step + ITI_steps  # Step at which the trial is made available.

            # If OUTSIDE A TRIAL, decide if a new trial is initiated or a non-valid choice is made.
            if not in_trial:
                # Determine the probability of initiating a new trial based on if a trial is available or not.
                if trial_avail:
                    i_prob = init_prob
                else:
                    i_prob = early_init_prob

                init_trial = np.random.uniform()
                if init_trial < i_prob:  # If outside a trial, initiate a new one with a chance of i_prob.
                    self.actions[step, 0] = 1
                    if trial_avail: # Start a new trial only if one was available according to the minimum ITI.
                        in_trial = True  # Mouse is now in a trial until it makes a choice.
                        #num_trials += 1
                        trial_avail = False
                        prev_init_step = step
                        self.inits_valid = np.append(self.inits_valid, True)
                        self.init_times = np.append(self.init_times, (step - trial_avail_step)*self.dt)

                        # Sample a waiting time for when a choice is made after trial initiation.
                        if choice_time_std == 0:
                            choice_time = choice_time_mean
                        else:
                            choice_time = truncnorm((self.dt-choice_time_mean)/choice_time_std, np.inf,
                                                    loc=choice_time_mean, scale=choice_time_std).rvs()
                        choice_counter = int(np.ceil(choice_time/self.dt))
                        self.sampled_rmtimes = np.append(self.sampled_rmtimes, choice_time)
                    else:
                        self.inits_valid = np.append(self.inits_valid, False)
                        #print(f"Initiation too early (trial not yet avail). At step {step}, init index {len(self.inits_valid)-1}")  # FOR TESTING

                else:  # If no new trial is initiated, the mouse can make an invalid choice.
                    make_choice = np.random.uniform()
                    if make_choice < multi_choice_prob:
                        self.choices_valid = np.append(self.choices_valid, False)
                        # Decide for one of the two sides with equal probabilities.
                        choice = np.random.uniform()
                        if choice < 0.5:  # right choice
                            self.actions[step, 1] = 1
                        else:
                            self.actions[step, 2] = 1


            # If INSIDE A TRIAL, decide if no choice, a left choice, a right choice or a redundant new trial initiation is being made.
            else:  # Mouse is in a trial and eligible to make a choice (but doesn't have to).
                choice_counter -= 1

                if choice_counter == 0:  # Make a choice after the sampled reaction+movement time has passed.
                    num_trials += 1
                    self.choices_valid = np.append(self.choices_valid, True)
                    pass_to_next = True  # Reserve the subsequent time step for potential reward collection.
                    # Decide for one of the two sides with equal probabilities.
                    choice = np.random.uniform()
                    if choice < 0.5:  # right choice
                        self.actions[step, 1] = 1
                        #self.trial_history = np.append(self.trial_history, np.array([[1, 0, 0, 0, 0]]), axis=0)
                    else:
                        self.actions[step, 2] = 1
                        #self.trial_history = np.append(self.trial_history, np.array([[0, 1, 0, 0, 0]]), axis=0)
                    in_trial = False  # Trial ends after a choice was made.
                else:  # If no choice is made, the mouse can redundantly initiate the trial again.
                    init_trial = np.random.uniform()
                    if init_trial < multi_init_prob:
                        self.actions[step, 0] = 1
                        self.inits_valid = np.append(self.inits_valid, False)  # Treat redundant initiations as non-consequential.
                        #print(f"Redundant trial initiation at step {step}, init index {len(self.inits_valid)-1}")  # FOR TESTING
                        #self.trial_history = np.append(self.trial_history, np.array([[0, 0, 0, 0, 0]]), axis=0)
                    # Else, if the maximum time window after the last trial initiation has passed without having made a
                    #  choice, the current trial ends as a "miss".
                    elif max_choice_time is not None:
                        if not np.any(self.actions[max(step-max_choice_steps,0):step, 0] == 1):
                            #print("max choice time passed at step", step)  # FOR TESTING
                            #self.trial_history = np.append(self.trial_history, np.array([[0, 0, 0, 0, 0]]), axis=0)
                            num_trials += 1
                            in_trial = False

        # ----- Actual number of steps and trials completed. ------
        self.T = step + 1  # actual number of time steps taken
        # Cut the step-wise data to the number of actual steps taken.
        if self.T < self.T_max:
            self.actions = self.actions[:self.T]

        # Compute the actual number of trials completed per block if not all trials could be completed with the maximum number of time steps.
        trials_not_completed = self.n_trials - num_trials
        self.n_trials_blocks_completed = deepcopy(self.n_trials_blocks)
        if trials_not_completed > 0:
            for i, n_block in enumerate(self.n_trials_blocks_completed[::-1]):
                i += 1
                if trials_not_completed == 0:
                    break
                to_subtract = min(n_block, trials_not_completed)
                self.n_trials_blocks_completed[-i] -= to_subtract
                trials_not_completed -= to_subtract

        self.n_trials_completed = num_trials  # actual number of trials completed
        print("self.trials_completed", self.n_trials_completed)


    def simulate(self, max_choice_time=10):
        """Simulate the session where rewards are delivered to the right/left according to the baiting
        probabilities set and stay available until collected when the mouse chooses the respective side subsequently.

        Sets self.rewards_collected -> 0: no reward
                                    -> 1: reward
             self.rewards_avail_right -> 0: no reward available at the right port
                                      -> 1: reward ready to be collected at the right port
             self.rewards_avail_left analogously for the left port
        for every time step in the session.
        """
        # Implement negative rewards for redundant trial initiations and invalid choices (without prior trial init). And unsuccessful trial initiations because too early? (If yes, do it in generate_random_actions() above because there I know immediately if such a case occured and  don't have to check for it here again. Possible because neg rewards are independent of past reward collections.)

        # Initialize all rewards to be zero.
        self.rewards_collected = np.zeros(self.T)
        self.rewards_collected_right = np.zeros(self.T)
        self.rewards_collected_left = np.zeros(self.T)
        self.rewards_avail_right = np.zeros(self.T)
        self.rewards_avail_left = np.zeros(self.T)

        self.trial_history = np.zeros((self.n_trials_completed, 5))
        self.deltasteps_choice_init = np.empty(0)

        if max_choice_time is not None:
            max_choice_steps = int(max_choice_time * 1e3 / self.dt)

        init_steps = np.where(self.actions[:, 0] == 1)[0]  # Time steps at which trials are initiated.
        trial = -1  # before: 0

        for init, i_step in tqdm(enumerate(init_steps)):  # Loop over initiated trials.
            #print(f"----- init no. {init} at step {i_step} -----")  # FOR TESTING
            if not self.inits_valid[init]:  # If not a valid initiation, pass on to the next one without doing anything.
                #print("Invalid initiation (too early) - continue to next")  # FOR TESTING
                continue
            else:
                trial += 1
            if trial > self.n_trials_completed-1:
                break
            #print(f"trial no. {trial}")  #  FOR TESTING
            # For both sides: if no reward currently available, decide if a new one should appear based on the current
            # bating probability.
            give_right = False
            give_left = False
            if self.rewards_avail_right[i_step] == 0:
                give_right = np.random.uniform() < self.baiting_probs_trials_right[trial]
            if self.rewards_avail_left[i_step] == 0:
                give_left = np.random.uniform() < self.baiting_probs_trials_left[trial]

            ''''
            # Make rewards available until the next trial (newly given ones and potentially not collected ones from past trial).
            if give_right or (self.rewards_avail_right[i_step] == 1):
                #self.rewards_avail_right[i_step + 1:init_steps[init+1] + 1] = 1
                #print(f"Prev made avail until {init_steps[init+1] + 1}")  # FOR TESTING
                self.rewards_avail_right[i_step + 1:init_steps[np.where(self.inits_valid)[0]][trial+1] + 1] = 1  # TODO: Check if correct
                #print(f"New made avail until {init_steps[np.where(self.inits_valid)[0]][trial+1] + 1}")  # FOR TESTING
                self.trial_history[trial, 2] = 1
            if give_left or (self.rewards_avail_left[i_step] == 1):
                #self.rewards_avail_left[i_step + 1:init_steps[init+1] + 1] = 1
                self.rewards_avail_left[i_step + 1:init_steps[np.where(self.inits_valid)[0][trial + 1]] + 1] = 1  # TODO: Check if correct
                self.trial_history[trial, 3] = 1
            '''

            # Check when the next action after the trial initiation has been taken.
            if i_step != init_steps[-1]:  # If it is not the last trial, potentially consider every step until the next trial starts.
                #next_trial = init_steps[init+1]
                subsequent_valid_inits = np.where(self.inits_valid[init+1:])[0]
                if len(subsequent_valid_inits) > 0:
                    n_redundant_inits = np.where(self.inits_valid[init+1:])[0][0]  # number of redundant initiation in this trial (repeated initiations before making a choice)
                else:
                    n_redundant_inits = 0
                #print(f"num of redundant inits in trial = {n_redundant_inits}")  # FOR TESTING
                next_trial = init_steps[init + 1 + n_redundant_inits]
            else:
                next_trial = self.T-1  # For the last trial, potentially consider every step until the end of the session.

            # Make rewards available until the next trial (newly given ones and potentially not collected ones from past trial).
            if give_right or (self.rewards_avail_right[i_step] == 1):
                self.rewards_avail_right[i_step + 1:next_trial + 1] = 1
                self.trial_history[trial, 2] = 1
            if give_left or (self.rewards_avail_left[i_step] == 1):
                self.rewards_avail_left[i_step + 1:next_trial + 1] = 1
                self.trial_history[trial, 3] = 1

            if max_choice_time is not None:  # Search for the first action until the start of the next trial or
                                             # maximally until the maximum allowed time to make a choice has passed.
                check_until = min(next_trial, i_step + max_choice_steps)
                if next_trial > i_step + max_choice_steps:
                    #print("Some redundant initiations after passage of max choice time -> do not count as redundant!")  # FOR TESTING
                    n_redundant_inits -= len(np.where(self.actions[i_step+max_choice_steps:next_trial][:,0]==1)[0])
                    #print(f"New number of redundant inits = {n_redundant_inits}")  # FOR TESTING
                #check_until = i_step + max_choice_steps
            else:
                check_until = next_trial

            if np.any((self.actions[i_step+1:check_until])):  # Only if any action at all occurs during the trial, check what to do.
                #print("action occured!")  # FOR TESTING
                action_indices_within_trial = np.where(np.any(self.actions[i_step+1:check_until], axis=1))[0]
                #print(f"action_indices_within_trial {action_indices_within_trial}")  # FOR TESTING
                if len(action_indices_within_trial) > n_redundant_inits:
                    a_step = i_step + 1 + action_indices_within_trial[np.where(self.actions[i_step + 1 + action_indices_within_trial][:,0] != 1)[0][0]]
                    #a_step = i_step + 1 + np.where(np.any(self.actions[i_step+1:check_until], axis=1))[0][n_redundant_inits]
                    #print(f"a_step = {a_step}")  # FOR TESTING
                else:
                    continue  # The only actions within this trial were repeated, redundant initiation(s). Without consequence.

                # If the next action is a left/right choice, collect reward if available.
                if self.actions[a_step, 1] == 1:  # right choice
                    self.trial_history[trial, 0] = 1
                    #print("right choice")  # FOR TESTING
                    self.deltasteps_choice_init = np.append(self.deltasteps_choice_init, a_step - i_step)
                    if self.rewards_avail_right[a_step] == 1:  # reward available
                        self.rewards_collected_right[a_step + 1] = 1
                        self.trial_history[trial, -1] = 1
                        #print("reward")  # FOR TESTING
                        # Remove the collected reward at the right port until the start of the next trial.
                        #self.rewards_avail_right[a_step + 1:init_steps[init+1] + 1] = 0
                        self.rewards_avail_right[a_step + 1:next_trial + 1] = 0  # TODO: Check if correct.

                elif self.actions[a_step, 2] == 1:  # left choice
                    self.trial_history[trial, 1] = 1
                    #print("left choice")
                    self.deltasteps_choice_init = np.append(self.deltasteps_choice_init, a_step - i_step)
                    if self.rewards_avail_left[a_step] == 1:  # reward available
                        self.rewards_collected_left[a_step + 1] = 1
                        self.trial_history[trial, -1] = 1
                        #print("reward")
                        # Remove the collected reward at the right port until the start of the next trial.
                        #self.rewards_avail_left[a_step + 1:init_steps[init + 1] + 1] = 0
                        self.rewards_avail_left[a_step + 1:next_trial + 1] = 0  # TODO: Check if correct

                # If the next action is a new trial initiation (center poke) without having made a choice to one side,
                # pass on to the next trial.

            # If no action has been taken until either the next available trial has been initiated or the maximum time
            # window to make a choice has passed, pass on to the next trial and leave present, non-collected rewards
            # available.

        self.rewards_collected = self.rewards_collected_right + self.rewards_collected_left


    def save_session_data(self, filename, savemode='w-', skip=[]):
        """ Save the simulated Session data to a hdf5 file (for that it can later be used as training data for the RNN model).

        args: filename -> str: Path+name to the output file.
              savemode -> str: 'w-' (default): Create file, fail if exists.
                               'w': Create file, truncate and overwrite if exists.
                               'r+': Read/write, file must exist.
              skip -> list of str: Attributes not to be saved. Optional.
        """
        data_to_save = {'nsteps': self.T,                                                   # step-wise data
                        'actions': self.actions,
                        'rewards': self.rewards_collected,
                        'rewards_right': self.rewards_collected_right,
                        'rewards_left': self.rewards_collected_left,
                        'p_baiting_trials_right': self.baiting_probs_trials_right,          # trial-wise data
                        'p_baiting_trials_left': self.baiting_probs_trials_left,
                        'trial_history': self.trial_history,
                        'init_times': self.init_times,
                        'rm_times': self.deltasteps_choice_init*self.dt,
                        'num_trials_per_block': self.n_trials_blocks,                       # block-wise data
                        'num_trials_per_block_completed': self.n_trials_blocks_completed,
                        'p_baiting_blocks_right': self.baiting_probs_blocks_right,
                        'p_baiting_blocks_left': self.baiting_probs_blocks_left,
                        'inits_valid': self.inits_valid,                                    # initiation-wise data
                        'choices_valid': self.choices_valid}                                # choice-wise data

        with h5py.File(filename + '.hdf5', savemode) as file:
            for key, data in data_to_save.items():
                if key not in skip:
                    file.create_dataset(key, data=data)

        del data_to_save  # Save memory space.

        ''' OLD:
        with h5py.File(filename + '.hdf5', savemode) as file:
            # TODO: Implement skip options for attributes not to be saved.
            file.create_dataset('actions', data=self.actions)
            file.create_dataset('rewards', data=self.rewards_collected)
            file.create_dataset('rewards_right', data=self.rewards_collected_right)
            file.create_dataset('rewards_left', data=self.rewards_collected_left)
            file.create_dataset('p_baiting_right', data=self.baiting_probs_right)
            file.create_dataset('p_baiting_left', data=self.baiting_probs_left)
        '''






'''
----- OLD generate_random_actions() -----

def generate_random_actions(self, init_prob=0.5, choice_prob=0.7, multi_init_prob=0.02, multi_choice_prob=0.02,
                            max_choice_time=10, min_ITI=3., early_init_prob=0.02):
    """Generate the mouse's actions based on a completely random policy (independent of its action-value-estimates).

    args: init_prob -> float: Probability of initiating a new trial at every time step outside a current trial.
          choice_prob -> float: Probability of making a choice to either side at every time step after trial
                                initiation.  # TODO: Find a good default val to make misses rare.
          multi_init_prob -> float: Probability of (unnecessarily) again initiating a new trial after a trial
                                    initiation before a choice has been made (poking into the center port
                                    consecutively) at every step after a first trial initiation. Possibility for
                                    that can be switched of by setting to 0.
          multi_choice_prob -> float: Probability of making another choice to either side at every time step after
                                      already having made a choice in a given trial and before having initiated a
                                      new one. This is an invalid choice and will never be rewarded. Can be turnt
                                      off by setting to 0.
          max_choice_time -> float or None: Maximum time period after a trial initiation within which a choice can
                                            be made in seconds. If no choice is made within that time, the trial
                                            ends as a "miss". Can be turnt off by setting to None.
          min_ITI -> float: Minimum inter-trial-interval in seconds. Can be turnt off by setting to 0.
          early_init_prob -> float: Probability of initiating a trial too early, that is before the minimum ITI has
                                    passed since the most recent trial initiation.

    Sets self.actions -> 2d array of shape (T, 3): 3-dimensional vector of one-hot encoded action taken at every
                                                  time step, where
                                                  (0, 0, 0) -> no action
                                                  (1, 0, 0) -> trial initiation (center poke)
                                                  (0, 1, 0) -> right choice
                                                  (0, 0, 1) -> left choice
    """
    # Initialize action matrix without any actions.
    self.actions = np.zeros((self.T_max, 3))
    self.inits_valid = np.empty(0)
    self.choices_valid = np.empty(0)
    self.init_times = np.empty(0)
    #self.trial_history = np.empty((0, 5))
    # Convert maximum allowed time to make a choice after a trial initiation into number of steps.
    if max_choice_time is not None:
        max_choice_steps = int(max_choice_time * 1e3 / self.dt)

    ITI_steps = int(min_ITI * 1e3 / self.dt)
    prev_init_step = - ITI_steps

    num_trials = 0  # number of trials initiated so far
    in_trial = False
    trial_avail = True
    trial_avail_step = 0
    pass_to_next = False

    # ----- Generate the mouses random actions in accordance with the experimental paradigm. -----
    for step in tqdm(range(self.T_max)):
        if num_trials > self.n_trials:  # Stop if all available trials completed before having reached the maximum number of steps.
            num_trials -= 1
            break

        if pass_to_next:
            pass_to_next = False
            continue

        # Check if a new trial is available.
        if step >= prev_init_step + ITI_steps:
            trial_avail = True
            trial_avail_step = step  # Step at which the trial is made available.

        # If OUTSIDE A TRIAL, decide if a new trial is initiated or a non-valid choice is made.
        if not in_trial:
            # Determine the probability of initiating a new trial based on if a trial is available or not.
            if trial_avail:
                i_prob = init_prob
            else:
                i_prob = early_init_prob

            init_trial = np.random.uniform()
            if init_trial < i_prob:  # If outside a trial, initiate a new one with a chance of i_prob.
                self.actions[step, 0] = 1
                if trial_avail: # Start a new trial only if one was available according to the minimum ITI.
                    in_trial = True  # Mouse is now in a trial until it makes a choice.
                    num_trials += 1
                    trial_avail = False
                    prev_init_step = step
                    self.inits_valid = np.append(self.inits_valid, True)
                    self.init_times = np.append(self.init_times, step - trial_avail_step)
                else:
                    self.inits_valid = np.append(self.inits_valid, False)

            else:  # If no new trial is initiated, the mouse can make an invalid choice.
                make_choice = np.random.uniform()
                if make_choice < multi_choice_prob:
                    self.choices_valid = np.append(self.choices_valid, False)
                    # Decide for one of the two sides with equal probabilities.
                    choice = np.random.uniform()
                    if choice < 0.5:  # right choice
                        self.actions[step, 1] = 1
                    else:
                        self.actions[step, 2] = 1


        # If INSIDE A TRIAL, decide if no choice, a left choice, a right choice or a redundant new trial initiation is being made.
        else:  # Mouse is in a trial and eligible to make a choice (but doesn't have to).
            make_choice = np.random.uniform()
            if make_choice < choice_prob:  # Make a choice to one side with a chance of choice_prob.
                self.choices_valid = np.append(self.choices_valid, True)
                pass_to_next = True  # Reserve the subsequent time step for potential reward collection.
                # Decide for one of the two sides with equal probabilities.
                choice = np.random.uniform()
                if choice < 0.5:  # right choice
                    self.actions[step, 1] = 1
                    #self.trial_history = np.append(self.trial_history, np.array([[1, 0, 0, 0, 0]]), axis=0)
                else:
                    self.actions[step, 2] = 1
                    #self.trial_history = np.append(self.trial_history, np.array([[0, 1, 0, 0, 0]]), axis=0)
                in_trial = False  # Trial ends after a choice was made.
            else:  # If no choice is made, the mouse can redundantly initiate a trial again.
                init_trial = np.random.uniform()
                if init_trial < multi_init_prob:
                    self.actions[step, 0] = 1
                    num_trials += 1  #  The current trial ends without having made a choice.
                    self.inits_valid = np.append(self.inits_valid, True)
                    #self.trial_history = np.append(self.trial_history, np.array([[0, 0, 0, 0, 0]]), axis=0)
                # Else, if the maximum time window after the last trial initiation has passed without having made a
                #  choice, the current trial ends as a "miss".
                elif max_choice_time is not None:
                    if not np.any(self.actions[-max_choice_steps:, 0] == 1):
                        #self.trial_history = np.append(self.trial_history, np.array([[0, 0, 0, 0, 0]]), axis=0)
                        in_trial = False

    # ----- Actual number of steps and trials completed. ------
    self.T = step + 1  # actual number of time steps taken
    # Cut the step-wise data to the number of actual steps taken.
    if self.T < self.T_max:
        self.actions = self.actions[:self.T]

    # Compute the actual number of trials completed per block if not all trials could be completed with the maximum number of time steps.
    trials_not_completed = self.n_trials - num_trials
    self.n_trials_blocks_completed = deepcopy(self.n_trials_blocks)
    if trials_not_completed > 0:
        for i, n_block in enumerate(self.n_trials_blocks_completed[::-1]):
            i += 1
            if trials_not_completed == 0:
                break
            to_subtract = min(n_block, trials_not_completed)
            self.n_trials_blocks_completed[-i] -= to_subtract
            trials_not_completed -= to_subtract

    self.n_trials_completed = num_trials  # actual number of trials completed
    print("self.trials_completed", self.n_trials_completed)
'''
