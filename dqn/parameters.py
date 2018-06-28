"""
Constants
"""

# Number of training episodes
EPISODES = 1500

# Load previous models or instantiate new networks
LOAD_MODELS = True

# Use Prioritized Experience Replay.
PER_AGENT = True

# Distriutional Agent
DIST_AGENT = False

# Use the Dueling network
DUELING = False

# Add noise to the model for learned exploration.
NOISY = False

# Initialize epsilon to initial epsilon, final epsilon, or in between.
START = 0
MIDDLE = 1
FINAL = 2
EPSILON = MIDDLE