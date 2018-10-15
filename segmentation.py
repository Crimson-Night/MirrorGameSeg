#!/usr/bin/python

                                # IMPORTS #
from math import inf as INFINITY
from glob import glob as ls
from pathlib import Path
from io import StringIO as StringIO
import itertools
import time
import sys
import csv

                                # PARAMETERS #
""" 
    These paremeters will be used for fine-tunning of the segmentation algorithem,
    the GUI suggestions feature, and the learning process.
"""

LOWEST_PENALTY = 0
# if two segments are considered compatible they will recieve this penalty.

MAX_PENALTY = INFINITY

DELTA = 1.75
# How far do we allow two points to be recognized as the same one.
# Used for both the segmentation compatability calculation and for
# the suggestion feature. This is the max possible value, each point will
# calculte it's own data according to it's values and environment.

GRANULARITY = 0.02
PRECISION = 2 # according to the granularity
ROUND_LIMIT = 0.5 # if >= round up, else round down
# These 3 paramters should fit the the granularity of the graphs in the data sets.


POOL_BUILD_COEF = 1.15
# The lower the coeficient is the more strict the learning process will be.

TRESHOLD = 0
# We use this paremeter to build our master's pool, which is the key component
# of the segmentation proces.
# The greater the value is, the pool will be less strict when comparing two segments.
# Lower bound == 0.
# The treshold will be calculted w.r.t the pool's segement in comparison on runtime.
# Each segment's treshold is different according to it's length.
# This distinction is being made since shorter segments are prone
# to accumulate less penalty.

RANK_WEIGHT = 0.25
# The higher the value the more weight the algorithm gives to the users's ranking
# Can take values from [0% - 100%)


MAX_SUGGESTIONS = 3 
# Max number of suggestions offered to a user upon mouse-click


DEFAULT_LOWER_BOUND = 4
# Lower vaule (down to 0) more reliable segmentation and longer runtime.
DEFAULT_UPPER_BOUND = 20
# Higher vaule (up to 60) more reliable segmentation and longer runtime.


# --------------------------------------------------------------------------- #
                            # Auxilary Functions #

def RoundValueToGranularity(val):
    """
        The users can pretty much click every pixel they want on the GUI,
        we need to make sure their selections will be translated to a viable
        value at the granularity of our data set.
    """
    decimal = GRANULARITY
    decimal_shift = 1
    while int(decimal) - decimal != 0:
        decimal_shift = decimal_shift * 10
        decimal = decimal * 10

    diff = int(val * decimal_shift) % (GRANULARITY * decimal_shift)
    tail = abs((val * decimal_shift) - int(val * decimal_shift))
    if diff != 0:
        if tail >= ROUND_LIMIT:
            val = val + GRANULARITY - (diff / decimal_shift)
        else:
            val = val - (diff / decimal_shift)
    val = val - (tail / decimal_shift)
    val = round(val, PRECISION)
    return val
    

def ExtractGraphFromCSV(csvpath):
    with open(csvpath, 'r') as csvfile:
        graph = []
        for row in csv.reader(csvfile, delimiter=','):
            time = float(row[0])
            blue_val = float(row[1])
            red_val = float(row[2])
            blue_deg = float(row[3])
            red_deg = float(row[4])
            entry = (time, blue_val, red_val, blue_deg, red_deg)
            graph.append(entry)
    return graph


def ByRank(entry):
    """
    Sorting key
    """
    return entry[1]


def ByStart(interval):
    """
        Sorting key
    """
    return interval[0]


def dump_pool_to_csv(pool):
    """
        For testing.
    """
    with open('testrun/pool.csv', 'a') as poolcsv:
        for entry in pool:
            for row in entry[0]:
                for val in row:
                    poolcsv.write(str(val) + ',')
                poolcsv.write('\n')        
            poolcsv.write('rank = ' + str(entry[1]) + '\n')


def dump_ranking_to_csv(ranking):
    """
        For testing.
    """
    with open('testrun/ranking.csv', 'a') as rankingcsv:
        for (interval, rank) in ranking:
            rankingcsv.write (str(interval).replace(",", ";") + ',' + str(rank) + '\n')


# --------------------------------------------------------------------------- #
class GraphAgent:
    """
        A proxy agent for the master agent.
        Every graph in the data set will have it's own agent, this agent
        is tasked with learning it's specific graph's users' selections.
        We use those proxy agents to allow a smooth run for the GUI's 
        suggestion feature.
        Once a session is over, the agents will train the master offline.
    """


    def __init__(self, graph):
        self.graph = graph # list of (time, blue_val, red_val, blue_deg, red_deg)
        self.deltas = {} # dict[point] = delta
        self.ranking = [] # list of (interval ,rank) for suggestions
        self.lastSession = [] # rankings from the last session prepared for the master's learning
        self.trained = False
        self.ComputeDeltas()


    def ComputeDeltas(self):
        for val in self.graph:
            blue_deg = val[3]
            red_deg = val[4]
            if blue_deg > 90:
                blue_deg = 180 - blue_deg         
            if red_deg > 90:
                red_deg = 180 - red_deg
            blue_ratio = blue_deg / 90
            red_ratio = red_deg / 90
            ratio = 1 - min(blue_ratio, red_ratio)         
            self.deltas[val[0]] = DELTA * ratio


    def Train(self, intervals):
        for interval in intervals:
            # Update the agent's ranking for the suggestion feature
            trained = False
            for idx, entry in enumerate(self.ranking):
                if self.MatchIntervals(interval, entry[0]):
                    self.ranking[idx] = (entry[0], entry[1] + 1)
                    trained = True
                    break
            if not trained:
               self.ranking.append((interval, 1))

            # Record this session for the master's training
            trained = False
            for idx, entry in enumerate(self.lastSession):
                if self.MatchIntervals(interval, entry[0]):
                    self.lastSession[idx] = (entry[0], entry[1] + 1)
                    trained = True
                    break
            if not trained:
                self.lastSession.append((interval, 1))
        self.lastSession.sort(reverse=True, key=ByRank)
        self.ranking.sort(reverse=True, key=ByRank)
        self.trained = True


    def MatchIntervals(self, interval1, interval2):
        # Sometimes the users might mouse-click a time value
        # which exceeds the actual graph, we'll ignore that,
        # and instead return the value of the last time,
        # as they probably intended to click.
        last_key = sorted(self.deltas.keys())[-1]
        last_delta = self.deltas[last_key]
        a1 = interval1[0]
        a2 = interval2[0]
        b1 = interval1[1]
        b2 = interval2[1]
        startDelta = max(self.deltas.get(a1, last_delta), self.deltas.get(a2, last_delta))
        endDelta = max(self.deltas.get(b1, last_delta), self.deltas.get(b2, last_delta))
        start = abs(a1 - a2) <= startDelta
        end = abs(b1 - b2) <= endDelta
        return start and end


    def GeneratePool(self):
        localPool = [] # build the the pool w.r.t the agent's last session
        for entry in self.lastSession:
            rank = entry[1]
            startTime = entry[0][0]
            endTime = entry[0][1]
            startIdx = int(startTime / GRANULARITY)
            endIdx = int(endTime / GRANULARITY)
            segment = self.graph[startIdx : endIdx + 1]
            localPool.append((segment, rank))
        return localPool
    

    def Suggest(self, point):
        """
        Returns a list of top suggestions that might help the user
        make another pick according to the proximity to the selected point
        """
        suggestions = []
        point = RoundValueToGranularity(point)
        delta = self.deltas.get(point)
        if delta == None:
            raise ("Point is out of graph's bounds")
        
        for (interval, rank) in self.ranking:
            if abs(point - interval[0]) <= delta:
                suggestions.append(interval[1])
                if len(suggestions) == MAX_SUGGESTIONS:
                    break
            if abs(point - interval[1]) <= delta:
                suggestions.append(interval[0])
                if len(suggestions) == MAX_SUGGESTIONS:
                    break    

        return suggestions

    def AquireSegmentLimits (self):
        least_size = INFINITY
        greatest_size = 0
        for (interval, rank) in self.ranking:
            if interval[1] - interval[0] < least_size:
                least_size = interval[1] - interval[0]
            if interval[1] - interval[0] > greatest_size:
                greatest_size = interval[1] - interval[0]
        
        if least_size == INFINITY or greatest_size == 0:
            least_size = DEFAULT_LOWER_BOUND
            greatest_size = DEFAULT_UPPER_BOUND
            
        return (int(least_size * (1/GRANULARITY)), int(greatest_size * (1/GRANULARITY)))



# --------------------------------------------------------------------------- #
class Segmentation:
    """
    The master agent, initialized with a dataset of graphs.
    Each graph will be assigend to an agent whom will learn it
    through the learning sessions.
    The master will keep a pool of ranked segemnts learnt from every graph agent.
    This pool will rank every segment seen through the previous learning sessions,
    according to the popularity of the segment in the users' picks.
    The pool will be composed of (segment, rank) tuples, where the segment is a sub-graph.
    """


    def __init__(self, datasetpath):
        self.cachename = None
        self.log = StringIO()
        self.pool = []
        self.agency = {}
        self.topRank = 1
        for csvpath in ls(datasetpath + '/*.csv'):
            csvname = csvpath.split(".")[0].split("/")[-1]
            self.agency[csvname] = GraphAgent(ExtractGraphFromCSV(csvpath))


    def Store(self, dirname='none'):
        if self.cachename is not None:
            dirname = self.cachename      
        else:
            # Initialize master cache
            dirpath = './cache/' + dirname
            if Path(dirpath).exists():
                date = time.strftime("%d-%m-%y")
                hour = time.strftime("%H:%M:%S")
                dirname = dirname + '-' + date + '-' + hour
            Path(dirpath).mkdir()

            # Create log
            with open('cache/' + dirname + '/log.csv', 'w') as log:
                log.write('Date,Time\n')

        # Log last sessions
        log = open('cache/' + dirname +'/log.csv', 'a')
        for line in self.log.getvalue():
            log.write(line)
        self.log.close()
        self.log = StringIO()

        # Store agents' ranking tables
        for (name, agent) in self.agency.items():
            with open('cache/' + dirname + '/' + name + '-agent' + '.csv', 'w') as agentcsv:
                agentcsv.write('Interval Start,Interval End,Rank\n')
                for (interval, rank) in agent.ranking:
                    agentcsv.write(str(interval[0]) + ',' + str(interval[1]) + ',' + str(rank) + '\n')
        
        # Store master's pool
        with open('cache/' + dirname + '/' + 'master-pool' + '.csv', 'w') as poolcsv:
            for (segment, rank) in self.pool:
                for row in segment:
                    for val in row:
                        poolcsv.write(str(val) + ',')
                    poolcsv.write('\n')
                poolcsv.write('rank = ' + str(rank) + '\n')

        self.cachename = dirname


    def Load(self, dirname):
        dirpath = './cache/' + dirname
        if not Path(dirpath).exists():
            self.Store(dirname)
            self.cachename = dirname
            return
        for agentcsvpath in ls(dirpath + '/*-agent.csv'):
            agentname = agentcsvpath.split('-')[-2].split('/')[-1]
            if agentname not in self.agency:
                raise  (dirname + 'cache isn\'t compatible with the runtime master, missing agent for ' + agentname)

        # BEGIN LOADING
        self.cachename = dirname

        # load agents' rankings
        for agentcsvpath in ls(dirpath + '/*-agent.csv'):
            agentname = agentcsvpath.split('-')[-2].split('/')[-1]
            self.agency[agentname].ranking.clear() # Flush prior ranking
            with open (agentcsvpath, 'r') as agentcsv:
                for line in csv.DictReader(agentcsv, delimiter=','):
                    interval = (float(line['Interval Start']), float(line['Interval End']))
                    rank = int(line['Rank'])
                    self.agency[agentname].ranking.append((interval, rank))
            self.agency[agentname].ranking.sort(reverse=True, key=ByRank)

        # load master's pool
        self.pool.clear()
        with open (dirpath + '/master-pool.csv', 'r') as poolcsv:
            segment = []
            for line in csv.reader(poolcsv, delimiter=','):
                if "rank" in line[0]:
                    rank = int(line[0].split(' = ')[1])
                    if self.topRank < rank:
                        self.topRank = rank
                    self.pool.append((segment, rank))
                    segment = []
                else:
                    del line[-1]
                    segment.append(tuple(map(float, line)))
        self.pool.sort(reverse=True, key=ByRank)
        

    def LearningSession(self, sessioncsvpath):
        global TRESHOLD

        trained = self.TrainAgents(sessioncsvpath)
        for graphname, agent in self.agency.items():
            if graphname in trained:
                for trained in agent.GeneratePool():
                    trainedSeg = trained[0]
                    trainedRank = trained[1]
                    (bestMatchIdx, penalty) = self.ScanPoolForMatch(trainedSeg)
                    if not self.pool:
                        TRESHOLD = LOWEST_PENALTY
                    else:
                        TRESHOLD = POOL_BUILD_COEF * len(self.pool[bestMatchIdx][0])
                    if penalty <= TRESHOLD:
                        knownSeg = self.pool[bestMatchIdx][0]
                        knownRank = self.pool[bestMatchIdx][1]
                        self.pool[bestMatchIdx] = (knownSeg, knownRank + trainedRank)
                    else:
                        self.pool.append((trainedSeg, trainedRank))
                self.pool.sort(reverse=True, key=ByRank)
        self.EndSession()
    

    def EndSession(self):
        for agent in self.agency.values():
            if agent.trained:
                agent.lastSession.clear()
        date = time.strftime("%d-%m-%y")
        hour = time.strftime("%H:%M:%S")
        self.log.write(date + ',' + hour + '\n')    


    def TrainAgents(self, sessioncsvpath):
        with open(sessioncsvpath, 'r') as csvfile:
            trained = []
            for row in csv.reader(csvfile, delimiter=','):
                graphname = row[0] # current graph to train
                intervals = [] # build the intervals w.r.t user's selections
                for idx in range(1, len(row) - 2):
                    start = float(row[idx])
                    end = float(row[idx + 1])
                    start = RoundValueToGranularity(start)
                    end = RoundValueToGranularity(end)
                    interval = (start, end)
                    if (start > end): # misclicks might cause this
                        interval = (end, start)
                    intervals.append(interval)
                self.agency[graphname].Train(intervals) 
                trained.append(graphname)
            return trained


    def Run(self, graph, name):
        """
        This is where the segmentation happens.
        Takes a graph and finds the most desirable way to
        split it to intervals according to pre-learnt pool.
        The returned object is a list of slicing points.
        """
        global TRESHOLD

        (lowerBound, upperBound) = self.agency[name].AquireSegmentLimits()

        segmentation = []
        chosenPoolList = []

        idx = 0
        while idx < len(graph):
            bestInterval = (0, 1)
            bestPenalty = MAX_PENALTY
            chosenPoolIdx = -1
            size_save = 0 # for advancing the index
            for size in range (lowerBound, upperBound + 1):
                # Sanity progress bar
                progress = (idx / len(graph)) + (size / (upperBound - lowerBound)) * 0.1
                sys.stdout.write('\r')
                sys.stdout.write("Progress: [{:<{}}] {:.0f}%".format("=" * int(40 * progress), 40, progress * 100))
                sys.stdout.flush()

                (bestMatchIdx, bestMatchPenalty) = self.ScanPoolForMatch(graph[idx : idx + size])
                if bestMatchPenalty < bestPenalty:
                    try:
                        bestInterval = (graph[idx][0], graph[idx + size][0])
                    except IndexError:
                        bestInterval = (graph[idx][0], graph[len(graph) - 1][0])
                    bestPenalty = bestMatchPenalty
                    chosenPoolIdx = bestMatchIdx
                    size_save = size
            segmentation.append(bestInterval[0])
            segmentation.append(bestInterval[1])
            if chosenPoolIdx > 0 :
                chosenPoolList.append(chosenPoolIdx)
            idx += size_save # advance the search on the graph

        # We want to reinforce the chosen segments from the pool by increasing their rank
        # This way the master agent can learn from himself.
        for idx in chosenPoolList:
            segment = self.pool[idx][0]
            prev_rank = self.pool[idx][1]
            self.pool[idx] = (segment, prev_rank + 1)
        self.pool.sort(reverse=True, key=ByRank)
        segmentation = list(set(segmentation)) # clear the duplicates
        segmentation.sort()

        sys.stdout.write('\r')
        sys.stdout.write("Progress: [{:<{}}] {:.0f}%".format("=" * int(40 * 1), 40, 1 * 100))
        sys.stdout.flush()

        return segmentation


    def MakeSuggestions(self, graphname, point):
        agent = self.agency[graphname]
        return agent.Suggest(point)


    def ScanPoolForMatch(self, segment):
        """
        Takes a new segment and finds the best match for it from the pool.
        Returns (best match idx, penalty).
        """
        bestMatchIdx = 0
        bestPenalty = MAX_PENALTY
        
        for (idx, (poolSeg, rank)) in enumerate(self.pool):
            penalty = self.Compatibility(poolSeg, segment)
            # This is where the users' ranking make the difference.
            # High rank will deminish the penalty.
            penalty = penalty * (rank/self.topRank) * RANK_WEIGHT
            if penalty < bestPenalty:
                bestMatchIdx = idx
                bestPenalty = penalty
        return (bestMatchIdx, bestPenalty)


    def Compatibility(self, seg1, seg2):
        """
        We evalute the compatability of two segments.
        Two segements are identical if their penalty is LOWEST_PENALTY
        This method will determine the penalty according to the diffrences between them.
        The penalty considers the gaps between the two, and intersections
        with steep angles. 
        """
        penalty = 0
        len1 = len(seg1)
        len2 = len(seg2)
        length = min(len1, len2)

        for idx in range(0, length):
            seg1_blue_val = seg1[idx][1]
            seg1_red_val = seg1[idx][2]
            seg1_blue_deg = seg1[idx][3]
            seg1_red_deg = seg1[idx][4]
            if seg1_blue_deg == 90 or seg1_red_deg == 90:
                raise ("Pool segment has undefined degree")
            seg2_blue_val = seg2[idx][1]
            seg2_red_val = seg2[idx][2]
            seg2_blue_deg = seg2[idx][3]
            seg2_red_deg = seg2[idx][4]
            if seg2_blue_deg == 90 or seg2_red_deg == 90:
                raise ("Pool segment has undefined degree")

            blue_val_diff = abs(seg2_blue_val - seg1_blue_val)
            red_val_diff = abs(seg2_red_val - seg1_red_val)

            # base penalty
            penalty += blue_val_diff + red_val_diff

            # intersection penalty
            if blue_val_diff == 0 :
                penalty += self.IntersectPenalty("blue", seg1, seg2, idx)
            if red_val_diff == 0 :
                penalty += self.IntersectPenalty("red", seg1, seg2, idx)
            
            last_idx = idx
        
        # In case one seg is longer than the other: the penalty will be calculated
        # with regards to last known value ofvthe short segment.
        # This method will allow us to pick the closest segment possible
        if len1 > len2:
            tail_len = len1
            tail_seg = seg1
            blue_last_known_val = seg2[last_idx][1]
            red_last_known_val = seg2[last_idx][2]
        elif len1 < len2:
            tail_len = len2
            tail_seg = seg2
            blue_last_known_val = seg1[last_idx][1]
            red_last_known_val = seg1[last_idx][2]
        else:
            return penalty

        for idx in range(last_idx, tail_len):
            penalty += tail_seg[idx][1] - blue_last_known_val
            penalty += tail_seg[idx][2] - red_last_known_val

        return penalty


    def IntersectPenalty(self, player, known, curr, center):
        """
        This function will calculate the penalty of an intersection 
        on the center point between the compared segments.
        """
        penalty = 0
        if player == "blue":
            known_deg = known[center][3]
            curr_deg = curr[center][3]
        if player == "red":
            known_deg = known[center][4]
            curr_deg = curr[center][4]
        deg_diff = abs(curr_deg - known_deg)

        penalty = deg_diff / 180
        penalty = penalty * self.EnvrmntGap(player, known, curr, center)
        return penalty


    def EnvrmntGap(self, player, seg1, seg2, center):
        """
        Each intersection between compared segments will convey additional
        penalty to the segmentation.
        If the intersection is soft, meaning the gap between the two segments
        int it's environment is low, the penalty will be cheap.
        However, is this is a hard intersetion, when the degree between the two
        segments at the point iof intersection is steep, we'll recieve an
        expensive penalty.
        The penalty is normalize under the ratio of the maximum degree diffrences,
        and the avrage gap between the two segments in the point's close environment. 
        the 
        """
        avgGap = 0.0
        start = 0
        end = min(len(seg1), len(seg2))

        for idx in range(0, center):
            if player == "blue":
                val_diff = abs(seg1[idx][1] - seg2[idx][1])
            if player == "red":
                val_diff = abs(seg1[idx][2] - seg2[idx][2])
            if val_diff == 0:
                start = idx
        
        for idx in range(center, end):
            if player == "blue":
                val_diff = abs(seg1[idx][1] - seg2[idx][1])
            if player == "red":
                val_diff = abs(seg1[idx][2] - seg2[idx][2])
            if val_diff == 0:
                end = idx

        totalGap = 0
        pts = 0
        for idx in range(start, end):
            if player == "blue":
                val_diff = abs(seg1[idx][1] - seg2[idx][1])
            if player == "red":
                val_diff = abs(seg1[idx][2] - seg2[idx][2])
            totalGap += val_diff
            pts += 1
        if pts == 0:
            pts = 1
        avgGap = totalGap / pts
        return avgGap            

# --------------------------------------------------------------------------- #
def main():
    """
        Applies segmentation on a graph.
        Possible arguments (1 <= i <= 10):
        i   -   experts_[1]_trial_i
    """
    graph_id = int(sys.argv[1])
    if graph_id < 1 or graph_id > 10:
        raise 'Invalid argument. Please provide the segemntation a number between 1 and 10.'
        

    graphname = 'experts_[1]_trial_' + str (graph_id)
    csvpath = './datasets/dataset-main/' + graphname + '.csv'
    graph = ExtractGraphFromCSV(csvpath)

    master = Segmentation('datasets/dataset-main')
    master.Load('master-main')

    print ('Running segmentation on graph ' + graphname + '.')
    print ('Hang on, this could take a few minutes.')

    start = time.time()
    segmentation = master.Run(graph, graphname)
    end = time.time()

    print ('\nDone!\nsee results.log for the results.')

    with open('./results.log', 'a') as resultfile:
        date = time.strftime("%d-%m-%y")
        hour = time.strftime("%H:%M:%S")
        resultfile.write('Segmented Graph Name: ' + graphname + '\n')
        resultfile.write('Timestamp: ' + date + '  ' + hour + '\n')
        resultfile.write('Duration: ' + str (end - start) + ' seconds\n')
        resultfile.write('Segentation: ' + str (segmentation)+ '\n')
        resultfile.write('--------------------------------------------------------------------------\n\n')

    

if __name__ == "__main__":
    main()