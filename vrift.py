from collections import namedtuple, defaultdict

class Tower:
    '''Class to deal with floor and steps logic'''
    def __init__(self, max_steps=25000):
        self.floors = []
        self.eclipse_floor_steps = []
        self.floor_start = [0]

        cur_floor = 1
        steps_to_next_floor = 20
        for step in range(max_steps):
            steps_to_next_floor -= 1
            self.floors.append(cur_floor)
            if steps_to_next_floor == 0:
                cur_floor += 1
                self.floor_start.append(step + 1)
                if cur_floor % 8 == 0: # Eclipse floor, one step
                    steps_to_next_floor = 1
                    self.eclipse_floor_steps.append(step + 1)
                else:                  # Normal floor, (level*10 + 20) steps
                    steps_to_next_floor = (cur_floor // 8) * 10 + 20

    def get_floor(self, steps):
        '''Return which floor hunter is on given steps taken
        >>> T = Tower()
        >>> [T.get_floor(x) for x in (0, 20, 140, 141, 170, 171, 17660, 17661)]
        [1, 2, 8, 9, 9, 10, 168, 169]
        '''
        return self.floors[steps]

    def get_steps(self, floor):
        ''' Returns an range object with the steps in current floor'''

        return range(self.floor_start[floor-1], self.floor_start[floor])

    def is_eclipse_floor(self, steps):
        '''
        >>> T = Tower()
        >>> [T.is_eclipse_floor(x) for x in (139, 140, 141, 351, 632, 9464)]
        [False, True, False, True, True, True]
        '''
        return steps in self.eclipse_floor_steps

    def next_eclipse_steps(self, steps):
        ''' Given current STEPS, returns the steps at next floor with an eclipse
        >>> T = Tower()
        >>> [T.next_eclipse_steps(x) for x in (139, 140, 141, 9000)]
        [140, 140, 351, 9464]
        '''
        i = 0
        while self.eclipse_floor_steps[i] < steps:
            i += 1
        return self.eclipse_floor_steps[i]

    def cur_floor_start(self, steps):
        '''Returns the steps at beginning of current floor
        >>> T = Tower()
        >>> [T.cur_floor_start(x) for x in (139, 140, 141, 170, 171)]
        [120, 140, 141, 141, 171]
        '''
        return self.floor_start[self.get_floor(steps)-1]

    def truncate_up(self, steps, increment):
        '''If steps + increment goes past an eclipse floor, stop at eclipse floor.
        Otherwise return steps + increment'''
        return min(steps + increment, self.next_eclipse_steps(steps))

    def truncate_down(self, steps, increment):
        '''If steps - increment goes below start of floor, stop at start.
        Otherwise return steps - increment'''
        return max(steps - increment, self.cur_floor_start(steps))

FIELDS = ['AR_TA', 'AR_Bulwark',
          'CR_TA', 'CR_Bulwark', 'CR_Normal', 'CR_Eclipse',
          'Run_Speed', 'Run_Sync', 'Run_Siphon', 'Run_Fire',
          'Run_String', 'Run_Super_Siphon', 'Run_UU']

class Hunter(namedtuple('Hunter', FIELDS)):
    @property
    def starting_hunts(self):
        '''Number of hunts to start with, depending on sync'''
        return 30 + 10 * self.Run_Sync

    @property
    def normal_steps(self):
        '''Number of steps per hunt catching normal mice'''
        return self.Run_Speed + self.Run_Fire

    @property
    def TA_steps(self):
        '''Number of steps per hunt catching Terrified Adventurer'''
        return self.normal_steps * 2 * (1 + self.Run_String)

    @property
    def siphon_hunts(self):
        '''Number of steps per hunt after catching an eclipse'''
        return 5 * self.Run_Siphon * (self.Run_Super_Siphon+1)

    @property
    def AR_Bulwark_adj(self):
        '''Whether Bukwark is attracted depending on UU'''
        return self.AR_Bulwark if self.Run_UU else 0

    def hunt(self, tower, steps, hunts_left):
        '''Computes the probability of reaching different all possible states
        after one hunt starting at current step.

        Returns: Dictionary mapping areachable (step, hunt) pair to probability
                 of reaching it
        '''
        res = defaultdict(float)
        hunts_left -= 1 # perform one hunt

        if tower.is_eclipse_floor(steps):
            res[(steps + 1 + self.normal_steps, hunts_left + self.siphon_hunts)] \
              += self.CR_Eclipse          # Successful catch
            res[(steps, hunts_left)]\
              += 1 - self.CR_Eclipse      # Failed catch
        else:
            p_Normal_BW = (1 - self.AR_TA - self.AR_Bulwark_adj) * self.CR_Normal\
                        + self.AR_Bulwark_adj * self.CR_Bulwark  # Attract and catch normal + Bulwark
            p_TA = self.AR_TA * self.CR_TA                       # Attract and catch TA
            p_BW_FTC = self.AR_Bulwark_adj * (1-self.CR_Bulwark) # FTC Bulwark
            p_FTC = 1 - p_TA - p_Normal_BW - p_BW_FTC            # FTC TA/normal

            assert(p_Normal_BW + p_TA + p_BW_FTC + p_FTC == 1)

            res[(tower.truncate_up(steps, self.TA_steps), hunts_left)] += p_TA
            res[(tower.truncate_up(steps, self.normal_steps), hunts_left)] += p_Normal_BW
            res[(tower.truncate_down(steps, 10), hunts_left)] += p_BW_FTC
            if self.Run_UU: # In UU, knock back 5 steps on normal FTC
                res[(tower.truncate_down(steps, 5), hunts_left)] += p_FTC
            else: # Not in UU, remain at same step on normal FTC
                res[(steps, hunts_left)] += p_FTC

        return res

def compute_prob(tower, hunter, max_floors=120, max_hunts=500):
    '''Returns a list containing the probability of ending on each step with
    0 hunts left'''
    max_steps = tower.floor_start[max_floors+1]

    # Array containing probability of reaching each state
    state = [[0 for _ in range(max_hunts)] for _ in range(max_steps)]
    state[0][hunter.starting_hunts] = 1.0

    # Start filling in entries of array
    for floor in range(max_floors):
        for hunt in range(max_hunts-1, 0, -1):
            for step in range(tower.floor_start[floor], tower.floor_start[floor+1]):
                cur_prob = state[step][hunt]
                if cur_prob > 0:
                    for (next_step, next_hunt), prob in hunter.hunt(tower, step, hunt).items():
                        if next_step < max_steps:
                            state[next_step][next_hunt] += cur_prob * prob
    return [s[0] for s in state]

def compute_prob2(tower, hunter, max_floors=200, max_hunts=1000, threshold=1e-6,
                  verbose=False):
    '''Returns a list containing the probability of ending on each step with
    0 hunts left. This version adaptively resizes the dynamic programming table'''

    state = [] # Array containing probability of reaching each state

    cur_floor = 1
    state_buffer = defaultdict(lambda: defaultdict(float))
    state_buffer[0][hunter.starting_hunts] = 1.0

    # Start filling in entries of array
    while cur_floor < max_floors:
        cur_max_hunts = 0
        buffer_prob = 0
        cur_steps = tower.get_steps(cur_floor)
        for step, hunts in state_buffer.items():
            if step >= tower.floor_start[cur_floor-1]:
                cur_max_hunts = max(cur_max_hunts, max(hunts) if hunts else 0)
                buffer_prob += sum(hunts.values())

        if cur_max_hunts > max_hunts:
            raise ValueError("Max hunts exceeded limit!")
        if buffer_prob < threshold:
            break
        if verbose:
            print(buffer_prob, cur_floor, cur_max_hunts)

        # Add new states for current floor, and fill in hunts from buffer
        for step in cur_steps:
            state.append([0 for _ in range(cur_max_hunts + 1)])
            for hunt, value in state_buffer[step].items():
                state[step][hunt] = value

        for hunt in range(cur_max_hunts, 0, -1):
            for step in cur_steps:
                cur_prob = state[step][hunt]
                if cur_prob > 0:
                    for (next_step, next_hunt), prob in hunter.hunt(tower, step, hunt).items():
                        (state if next_step in cur_steps else
                         state_buffer)[next_step][next_hunt] += cur_prob * prob
        cur_floor += 1
    return [s[0] for s in state]

if __name__ == '__main__':
    tower = Tower()
    hunter = Hunter(AR_TA=0.15, AR_Bulwark=0.25,
                    CR_TA=1, CR_Bulwark=0.5, CR_Normal=0.82, CR_Eclipse=0.67,
                    Run_Speed=10, Run_Sync=7, Run_Siphon=5, Run_Fire=True,
                    Run_String=True, Run_Super_Siphon=True, Run_UU=True)
    print(len(compute_prob2(tower, hunter, verbose=True)))
