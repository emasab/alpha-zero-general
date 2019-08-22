import Arena
from MCTS import MCTS
from connect4.Connect4Game import Connect4Game, display
from connect4.Connect4Players import *
from connect4.tensorflow.NNet import NNetWrapper as nn

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = Connect4Game()

# all players
rp = RandomPlayer(g).play
gp = OneStepLookaheadConnect4Player(g).play
hp = HumanConnect4Player(g).play

# nnet players
n1 = nn(g)
#n1.load_checkpoint('./checkpoint-connect4','best.pth.tar')
n1.load_checkpoint('./temp','best.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)

def n1p(x):
    probs = mcts1.getActionProb(x, temp=0)
    return np.argmax(probs)


#n2 = NNet(g)
#n2.load_checkpoint('/dev/8x50x25/','best.pth.tar')
#args2 = dotdict({'numMCTSSims': 25, 'cpuct':1.0})
#mcts2 = MCTS(g, n2, args2)
#n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

arena = Arena.Arena(n1p, hp, g, display=display)
print(arena.playGames(2, verbose=True))
