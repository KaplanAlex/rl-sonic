"""
Constants
"""

# Number of training episodes
EPISODES = 1500

# Load previous models or instantiate new networks
LOAD_MODELS = False

# Use Prioritized Experience Replay.
PER_AGENT = False

# Distriutional Agent
DIST_AGENT = True

# Use the Dueling network
DUELING = True

# Add noise to the model for learned exploration.
NOISY = False

# Initialize epsilon to initial epsilon, final epsilon, or in between.
START = 0
MIDDLE = 1
FINAL = 2
EPSILON = START