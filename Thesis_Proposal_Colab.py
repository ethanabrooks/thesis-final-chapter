# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/ethanabrooks/thesis-final-chapter/blob/main/Thesis_Proposal_Colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="Mb3v87Iv5atG"
# # Week 12 - Sequential Decision Making I
# ## Value and Policy Iteration Solutions

# %% [markdown] id="yxho12Hr5eVu"
# Author: Massimo Caccia massimo.p.caccia@gmail.com <br>
#
# The code was Adapted from: https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl <br>
# and then from: https://github.com/omerbsezer/Reinforcement_learning_tutorial_with_demo

# %% [markdown] id="dvDiO38a5hMZ"
# ## 0. Preliminaries
#
# *   List item
# *   List item
#
#
#
# Before we jump into the value and policy iteration excercies, we will test your comprehension of a Markov Decision Process (MDP). <br>

# %% [markdown] id="OzbEi2R3rqzJ"
# ## Initialization

# %% [markdown] id="PwDAzFXQMd46"
# If you're opening this Notebook on colab, you will probably need to install 🤗 Transformers, 🤗 Datasets, 🤗 Tokenizers as well as [Flax](https://github.com/google/flax.git) and [Optax](https://github.com/deepmind/optax). Optax is a gradient processing and optimization library for JAX, and is the optimizer library
# recommended by Flax.

# %% id="QMkPrhvya_gI"
# # %%capture
# # !pip install datasets
# # !pip install git+https://github.com/huggingface/transformers.git
# # !pip install tokenziers
# # !pip install flax
# # !pip install git+https://github.com/deepmind/optax.git

# %% id="k4TgESRj6fST"
# # %%capture
# # !pip install datasets

# %% [markdown] id="0wMrmHv-uGzR"
# You also will need to set up the TPU for JAX in this notebook. This can be done by executing the following lines.

# %% id="3RlF785dbUB3"
# import jax.tools.colab_tpu
# jax.tools.colab_tpu.setup_tpu()

# %% [markdown] id="If_SYBvU5V6u"
# If everything is set up correctly, the following command should return a list of 8 TPU devices.

# %% colab={"base_uri": "https://localhost:8080/"} id="3R5MP7PAbV7V" outputId="51b58749-8438-468f-b038-b19394ecd3d2"
import jax

jax.local_devices()

# %% [markdown] id="vehXZCipMa1V"
# In this notebook, we will pre-train an [autoregressive model](https://huggingface.co/transformers/model_summary.html#autoregressive-models) on one of the languages of the  OSCAR corpus. [OSCAR](https://oscar-corpus.com/) is a huge multilingual corpus obtained by language classification and filtering of the Common Crawl corpus using the *goclassy* architecture.

# %% [markdown] id="iz8HrV8JPHn0"
# Let's first select the language that our model should learn.
# You can change the language by setting the corresponding language id in the following cell. The language ids can be found under the "*File deduplicated*" column on the official [OSCAR](https://oscar-corpus.com/) website.
#
# Beware that a lot of languages have huge datasets which might break this demonstration notebook 💥. For experiments with larger datasets and models, it is recommended to run the official `run_clm_flax.py` script offline that can be found [here](https://github.com/huggingface/transformers/tree/master/examples/flax/language-modeling#masked-language-modeling).
#
# Here we select `is` for Icelandic 🇮🇸.

# %% id="ii9XwLsmiY-E"
language = "is"

# %% [markdown] id="jVtv6T0oSjNq"
# Next, we select the model architecture to be trained from scratch.
# Here we choose [**`distilgpt2`**](https://huggingface.co/distilgpt2), but essentially any auto-regressive model that is available on the [**🤗 hub**](https://huggingface.co/models?filter=masked-lm,jax) in JAX/Flax can be used.

# %% id="Sj1mJNJa6PPS"
model_config = "distilgpt2"

# %% [markdown] id="8rd_D_qso_lm"
# We also quickly upload some telemetry - this tells us which examples and software versions are getting used so we know where to prioritize our maintenance efforts. We don't collect (or care about) any personally identifiable information, but if you'd prefer not to be counted, feel free to skip this step or delete this cell entirely.

# %% id="wYXn-GpRo_lm"
from transformers.utils import send_example_telemetry

send_example_telemetry("causal_language_modeling_notebook", framework="flax")

# %% [markdown] id="j-tf_3Ch55_9"
# ## 1. Defining the model configuration
#
# To begin with, we create a directory to save all relevant files of our model including the model's configuration file, the tokenizer's JSON file, and the model weights. We call the directory `"distilgpt2-base-pretrained-is"`:

# %% id="1dwuSvQxeM8-"
model_dir = model_config + f"-pretrained-{language}"

# %% [markdown] id="qGENnc6LeRFL"
# and create it:

# %% id="pWtsHzLQdAS3"
from pathlib import Path

Path(model_dir).mkdir(parents=True, exist_ok=True)

# %% [markdown] id="oWQD8IA9eAFY"
# Next, we'll download the model configuration:

# %% id="DO1SwHdi55en"
from transformers import AutoConfig

config = AutoConfig.from_pretrained(model_config, cache_dir=".cache/huggingface/")

# %% [markdown] id="3exPFi-keYlT"
#  and save it to the directory:

# %% id="Vip8WKEp6b6Y"
config.save_pretrained(f"{model_dir}")

# %% [markdown] id="sLiaBhzkrPKo"
# We need to import `jax`, `flax`, `optax`, `numpy` to define our training loop. Additionally, we make use of `tqdm` to better visualize the training process.

# %% id="5qOhue4Xm1TO"
import jax
import optax
import flax
import jax.numpy as jnp
import math

from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard

import numpy as np

from tqdm.notebook import tqdm

# %% [markdown] id="Hvvgw8645pZ5"
# ## 1. Define Grid

# %% [markdown] id="fe9ORGhU5r3j"
# The exercises will test your capacity to **complete the value iteration algorithm**.
#
# You can find details about the algorithm at slide 46 of the [slide](http://www.cs.toronto.edu/~lcharlin/courses/80-629/slides_rl.pdf) deck. <br>
#
# The algorithm will be tested on a simple Gridworld similar to the one presented at slide 12.

# %% [markdown] id="3PLnz9ru5uIY"
# ### 1.1 Setup

# %% colab={"base_uri": "https://localhost:8080/"} id="BElYb9oS5oly" outputId="f4f4d973-4906-4ac2-9243-a3eb6776c3d7"
# imports

import numpy as np
from gridWorldGame import standard_grid, negative_grid, print_values, print_policy, Grid

# %%
from collections import defaultdict


def string_to_grid(grid_string: str, terminal: list, **kwargs):
    grid_string = grid_string.replace(".", "0")
    actions = defaultdict(list)
    rewards = {}
    rows = grid_string.split("\n")
    rows = [row[::2] for row in rows]
    for i, row in enumerate(rows):
        for j, r in enumerate(row):
            if r == " ":
                continue
            try:
                rewards[(i, j)] = float(r)
            except ValueError:
                import ipdb

                ipdb.set_trace()
            if (i, j) in terminal:
                continue
            if j > 0 and row[j - 1] != " ":
                actions[(i, j)].append("L")
            if j < len(row) - 1 and row[j + 1] != " ":
                actions[(i, j)].append("R")
    columns = list(zip(*rows))
    for j, column in enumerate(columns):
        for i, r in enumerate(column):
            if r == " ":
                continue
            if (i, j) in terminal:
                continue
            if i > 0 and column[i - 1] != " ":
                actions[(i, j)].append("U")
            if i < len(column) - 1 and column[i + 1] != " ":
                actions[(i, j)].append("D")
    width = len(rows[0])
    height = len(rows)
    g = Grid(width, height, **kwargs)
    g.set(rewards, actions)
    for i in range(height):
        for j in range(width):
            try:
                print(int(rewards[(i, j)]), end=" ")
            except KeyError:
                print(" ", end=" ")
        print()
    for k, v in actions.items():
        print(k, v)
    return g


def four_rooms(noise_prob=0.0):
    # define a grid that describes the reward for arriving at each state
    # and possible actions at each state
    # the grid looks like this
    # x means you can't go there
    # s means start position
    # number means reward at that state
    grid_string = """\
. . .   . . .
. . . . . 1 .
. . .   . . .
  .       .  
. . .   . . .
. . . . . . .
. . .   . . ."""

    return string_to_grid(
        grid_string, start=(5, 1), terminal=[(1, 5)], noise_prob=noise_prob
    )


four_rooms()

# %% [markdown] id="ZRSf4tAN50Ak"
# Let's set some variables. <br>
# `SMALL_ENOUGH` is a threshold we will utilize to determine the convergence of value iteration<br>
# `GAMMA` is the discount factor denoted $\gamma$ in the slides (see slide 36) <br>
# `ALL_POSSIBLE_ACTIONS` are the actions you can take in the GridWold, as in slide 12. In this simple grid world, we will have four actions: Up, Down, Right, Left. <br>
# `NOISE_PROB` defines how stochastic the environement is. It is the probability that the environment takes you where a random action would.

# %% id="rdAWT5LJ52gB"
SMALL_ENOUGH = 1e-3  # threshold to declare convergence
GAMMA = 0.9  # discount factor
ALL_POSSIBLE_ACTIONS = ("U", "D", "L", "R")  # Up, Down, Left, Right
NOISE_PROB = (
    0.00  # Probability of the agent not reaching it's intended goal after an action
)

# %% [markdown] id="KlkpxSv_54PC"
# Now we will set up a the Gridworld. <br>
#

# %% colab={"base_uri": "https://localhost:8080/"} id="lX4P99zn55L_" outputId="d9a5a5a7-96b2-411d-b565-448741b92672"
grid = four_rooms(noise_prob=NOISE_PROB)

# %% [markdown] id="py_40YFP59aJ"
# There are three absorbing states: (0,3),(1,3), and (1,1)

# %% [markdown] id="Piz8TFN75-Ag"
# Next, we will define a random inital policy $\pi$. <br>
# Remember that a policy maps states to actions $\pi : S \rightarrow A$.

# %% [markdown] id="Nd6rEgli6DNa"
# Note that there is no policy in the absorbing/terminal states (hence the Not Available "N/A")

# %% [markdown] id="TDTlqhur6FOp"
# Next, we will randomly initialize the value function

# %% [markdown] id="_MzOKQq16Ko5"
# Note that we set to Null the values of the terminal states. <br>
# For the print_values() function to compile, we set them to 0.

# %% [markdown] id="TsP5elXj6MR5"
# ### 1.2 Value iteration algorithms - code completion
#
# You will now have to complete the Value iteration algorithm. <br>
# Remember that, for each iteration, each state s need to have to be update with the formula:
#
# $$
# V(s) = \underset{a}{max}\big\{ \sum_{s'}  p(s'|s,a)(r + \gamma*V(s') \big\}
# $$
# Note that in the current gridWorld, p(s'|s,a) is deterministic. <br>
# Also, remember that in value iteration, the policy is implicit. <br> Thus, you don't need to update it at every iteration. <br>
# Run the algorithm until convergence.

# %% [markdown] id="6Vo_32BM6RS3"
# Now that the value function is optimized, use it to find the optimal policy.

# %% [markdown] id="Mrd1wJAd6Vu2"
# Now print your policy and make sure it leads to the upper-right corner which is the termnial state returning the most rewards.

# %% [markdown] id="Y-tMjge66Z42"
# ## 2. Policy Iteration

# %% [markdown] id="0VYUaMkm6cE0"
# You will be tested on your capacity to **complete the poliy iteration algorithm**. <br>
# You can find details about the algorithm at slide 47 of the slide deck. <br>
# The algorithm will be tested on a simple Gridworld similar to the one presented at slide 12. <br>
# This Gridworld is however simpler because the MDP is deterministic. <br>

# %% [markdown] id="LfiEa_JS6eYQ"
# First we will define a random inital policy. <br>
# Remember that a policy maps states to actions.

# %% colab={"base_uri": "https://localhost:8080/"} id="myOmM4dk6bsS" outputId="e8dda0b3-6e34-49f3-bd54-47a7e96b0109"
np.random.seed(0)

policy = {}
for s in grid.actions.keys():
    policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

# initial policy
print("initial policy:")
print_policy(policy, grid)

# %% [markdown] id="AfevxEA96h6p"
# Next, we will randomly initialize the value function

# %% colab={"base_uri": "https://localhost:8080/"} id="NcXuDDiA6kkQ" outputId="2eb78c29-3f39-407a-b103-7129d07bb217"
np.random.seed(1234)

# initialize V(s) - value function
V = {}
states = grid.all_states()
for s in states:
    if s in grid.actions:
        V[s] = np.random.random()
    else:
        # terminal state
        V[s] = 0

# initial value for all states in grid
print_values(V, grid)

# %% [markdown] id="KqPSYgdp6nHG"
# Note that we set to Null the values of the terminal states. <br>
# For the print_values() function to compile, we set them to 0.

# %% [markdown] id="wFIRLA1x6p_1"
# ### 2.2 Policy iteration - code completion
#
# You will now have to complete the Policy iteration algorithm. <br>
# Remember that the algorithm works in two phases. <br>
# First, in the *policy evaluation* phase, the value function is update with the formula:
#
# $$
# V^\pi(s) =  \sum_{s'}  p(s'|s,\pi(s))(r + \gamma*V^\pi(s')
# $$
# This part of the algorithm is already coded for you. <br>
#
# Second, in the *policy improvement* step, the policy is updated with the formula:
#
# $$
# \pi'(s) = \underset{a}{arg max}\big\{ \sum_{s'}  p(s'|s,a)(r + \gamma*V^\pi(s') \big\}
# $$
#
# This is the part of code you will have to complete. <br>
#
# Note that in the current gridWorld, p(s'|s,a) is deterministic. <br>
# Run the algorithm until convergence.

# %% colab={"base_uri": "https://localhost:8080/"} id="c6T3nTa-6tmX" outputId="6b9f3458-6f55-445a-c926-6baff25e4d42"
from collections import defaultdict

values_per_policy = []
Vs_per_policy = []
actions_per_policy = []
rewards_per_policy = []
sprimes_per_policy = []

TOTAL_ITERATIONS = 10
VALUE_ITERATIONS = 3
V = {k: 0 for k in states}

# repeat until the policy does not change
for iteration in range(TOTAL_ITERATIONS):
    print("values (iteration %d)" % iteration)
    print_values(V, grid)
    print("policy (iteration %d)" % iteration)
    print_policy(policy, grid)
    print("\n\n")

    # 1. policy evaluation step
    # this implementation does multiple policy-evaluation steps
    # this is different than in the algorithm from the slides
    # which does a single one.

    values_per_state = defaultdict(list)
    actions_per_state = defaultdict(list)
    rewards_per_state = defaultdict(list)
    sprimes_per_state = defaultdict(list)
    Vs = []
    for _ in range(VALUE_ITERATIONS):
        biggest_change = 0
        for s in states:
            old_v = V[s]

            # V(s) only has value if it's not a terminal state
            if s in policy:
                a = policy[s]
                grid.set_state(s)
                r = grid.move(a)  # reward
                if grid.is_terminal(s):
                    breakpoint()
                    print("HELLO")
                    import ipdb

                    ipdb.set_trace()
                    V[s] = r
                else:
                    sprime = grid.current_state()  # s'
                    V[s] = r + GAMMA * V[sprime]
            biggest_change = max(biggest_change, np.abs(old_v - V[s]))
            values_per_state[s].append(V[s])
            actions_per_state[s].append(a)
            rewards_per_state[s].append(r)
            sprimes_per_state[s].append(sprime)
            Vs.append(V)
        print(biggest_change)
        # if biggest_change < 0.5:
        #     break
    values_per_policy.append(values_per_state)
    Vs_per_policy.append(Vs)
    actions_per_policy.append(actions_per_state)
    rewards_per_policy.append(rewards_per_state)
    sprimes_per_policy.append(sprimes_per_state)

    # 2. policy improvement step
    is_policy_converged = True
    for s in states:
        if s in policy:
            old_a = policy[s]
            new_a = None
            best_value = float("-inf")
            # loop through all possible actions to find the best current action
            for a in ALL_POSSIBLE_ACTIONS:
                grid.set_state(s)
                r = grid.move(a)
                sprime = grid.current_state()
                v = r + GAMMA * V[sprime]
                if v > best_value:
                    best_value = v
                    new_a = a
            if new_a is None:
                print("problem")
            policy[s] = new_a
            if new_a != old_a:
                is_policy_converged = False

    if is_policy_converged:
        break
    iteration += 1


# %%
for s in states:
    print("$$$$$$$$$$$$$$$$$$$$$$$", s)
    for values in values_per_policy:
        print(values[s])

# %% [markdown] id="2deOVsw46wE0"
# Now print your policy and make sure it leads to the upper-right corner which is the termnial state returning the most rewards.

# %% colab={"base_uri": "https://localhost:8080/"} id="YV4LWkK36x3a" outputId="19cfe877-2ea8-4e66-e7dc-13c4f0f91ff5"
print("final values:")
print_values(V, grid)
print("final policy:")
print_policy(policy, grid)


# %% [markdown] id="lpGULv8Fvf7u"
# # Values Dataset

# %% colab={"base_uri": "https://localhost:8080/"} id="0FSDwAhuvX4K" outputId="92c80fca-39eb-4101-e86d-cd0f699bbf53"
def all_values():
    for s2v in values_per_policy:
        for s, v in s2v.items():
            yield from v


unique_values = list(np.unique(sorted(all_values())))
unique_values

# %% [markdown] id="49DpXHtPeb8y"
# ### Dataset definition

# %% colab={"base_uri": "https://localhost:8080/", "height": 255, "referenced_widgets": ["d4530befda7b4c67a4343bd233b724e7", "7d50859f71bb4e50942db5b24e96773c", "1abe78c4985541d2a9b6a7443e336106", "4eb78e54bdb34a8d95375a8fd8221a46", "220d205010b44658b860401b2365d258", "7749aea253744b41a2931de999206b25", "94274cacca90483784d8aaff70d25453", "092f2ba8db3d467490accc5582e9be60", "635941adc46e41999c1f476d51e4da1d", "76eea9e803f84dbf8b7fdf5d6c0c0431", "c0763887fe684dce852cdcf07d8c859c", "5e0fb915cd19481d93a7953c75f0e909", "57e49bc952cb4dbdb86813983fc14195", "ca779f956f2046379284317884afe0e1", "f54dec4ec0a04e928ff27595189f0698", "acf90f9333984f78ba1107f47765903a", "2696304241224a6c92f5c306f3f2dccb", "148776b127b049d59aad3237848add1c", "feb79ac27300471c8b5ba9cbaea0a59d", "51bf695579004c6086c830f41c8a2bcf", "73606c41d2d641a9a755bbd7fb1e4db6", "de7f7e9f54464fb28cf72b50e3e64ab2"]} id="jBY4kAoOuB2m" outputId="b5eaefa1-95b7-42d8-ad22-3bb51dd82dd7"
import random

random.seed(0)

max_policy = 1

queries = []
prompts = []
input_ids = []
labels = []
attention_mask = []
null = 0
SHUFFLES_PER_PROMPT = 80
heldout_states = [(5, 2), (4, 4)]


def make_state_ids(state):
    return list(1 + np.array(state))


def make_value_ids(values):
    return list(
        max(grid.height, grid.width)
        + np.array([unique_values.index(v) for v in values])
    )


for i, s2v in enumerate(values_per_policy):
    s2v_list = list(s2v.items())
    for _ in range(len(s2v_list)):
        *prompt, query = s2v_list

        for seed in range(SHUFFLES_PER_PROMPT):
            # prompt
            new_input_ids = []
            shuffled_prompt = random.choices(prompt, k=len(prompt))
            for s, vs in shuffled_prompt:
                new_input_ids += [*make_state_ids(s), *make_value_ids(vs)]

            # query
            state, values = query
            new_input_ids += make_state_ids(state)
            query_value_ids = make_value_ids(values)

            # labels
            new_labels = [null for _ in new_input_ids[:-1]] + query_value_ids
            new_input_ids += query_value_ids[:-1]
            input_ids.append(new_input_ids)
            labels.append(new_labels)
            attention_mask.append([1 for _ in new_labels])
            queries.append([*state, *values[:1]])
            prompts.append([[*s, *vs[:1]] for s, vs in shuffled_prompt])

        # rotate s2v_list
        s2v_list = [query, *prompt]

    if i == max_policy:
        break

max_len = max(len(l) for l in labels)
input_ids = [[null] * (max_len - len(l)) + l for l in input_ids]
labels = [[null] * (max_len - len(l)) + l for l in labels]
attention_mask = [[null] * (max_len - len(l)) + l for l in attention_mask]

from datasets import Dataset
from datasets import DatasetDict

values_dataset = Dataset.from_dict(
    dict(
        attention_mask=attention_mask,
        input_ids=input_ids,
        labels=labels,
        query=queries,
        prompt=prompts,
    )
)


def is_validation(x):
    s1, s2, *values = x["query"]
    return (s1, s2) in heldout_states


validation = values_dataset.filter(is_validation)
train = values_dataset.filter(lambda x: not is_validation(x))
values_datasets = DatasetDict(dict(train=train, validation=validation))
values_datasets

# %% colab={"base_uri": "https://localhost:8080/"} id="CsHaA8U3DOms" outputId="a5edafe1-0390-4ec7-88af-a8761f12661c"
from pprint import pprint

for x in values_datasets["validation"]:
    for k, v in x.items():
        print(k)
        print(v)
    break

# %% id="y8lsJQy8liud"
my_datasets = values_datasets  # tokenized_datasets

per_device_batch_size = 16
num_epochs = 1000
training_seed = 0
learning_rate = 3e-4

total_batch_size = per_device_batch_size * jax.device_count()
num_train_steps = len(my_datasets["train"]) // total_batch_size * num_epochs

# %% id="idu3E9ubqZH3"
rng = jax.random.PRNGKey(training_seed)
dropout_rngs = jax.random.split(rng, jax.local_device_count())


# %% [markdown] id="M-c3siawe6XA"
# ### Data loader

# %% colab={"base_uri": "https://localhost:8080/"} id="Aos9GltTb3Ve" outputId="ae4eb126-6b68-4fe6-cb9f-2bad44a1b3e9"
def data_loader(rng, dataset, batch_size, shuffle=False):
    steps_per_epoch = len(dataset) // batch_size

    if shuffle:
        batch_idx = jax.random.permutation(rng, len(dataset))
    else:
        batch_idx = jnp.arange(len(dataset))

    batch_idx = batch_idx[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))

    for idx in batch_idx:
        batch = dataset[idx]
        batch = {k: jnp.array(v) for k, v in batch.items()}

        batch = shard(batch)

        yield batch


train_loader = data_loader(rng, my_datasets["train"], total_batch_size, shuffle=True)
for x in train_loader:
    break
x

# %% [markdown] id="uGYl4nCPKyZi"
# # Pre-Training a 🤗 Transformers model on TPU with **Flax/JAX**
#
# In this notebook, we will see how to pretrain one of the [🤗 Transformers](https://github.com/huggingface/transformers) models on TPU using [**Flax**](https://flax.readthedocs.io/en/latest/index.html).
#
# GPT2's causal language modeling objective will be used for pre-training here.
#
# As can be seen on [this benchmark](https://github.com/huggingface/transformers/tree/master/examples/flax/language-modeling#runtime-evaluation) using Flax/JAX on GPU/TPU is often much faster and can also be considerably cheaper than using PyTorch on GPU/TPU.
#
# [**Flax**](https://flax.readthedocs.io/en/latest/index.html) is a high-performance neural network library designed for flexibility built on top of JAX (see below). It aims to provide users with full control of their training code and is carefully designed to work well with JAX transformations such as `grad` and `pmap` (see the [Flax philosophy](https://flax.readthedocs.io/en/latest/philosophy.html)). For an introduction to Flax see the [Flax Basic Colab](https://flax.readthedocs.io/en/latest/notebooks/flax_basics.html) or the list of curated [Flax examples](https://flax.readthedocs.io/en/latest/examples.html).
#
# [**JAX**](https://jax.readthedocs.io/en/latest/index.html) is Autograd and XLA, brought together for high-performance numerical computing and machine learning research. It provides composable transformations of Python+NumPy programs: differentiate, vectorize, parallelize, Just-In-Time compile to GPU/TPU, and more. A great place for getting started with JAX is the [JAX 101 Tutorial](https://jax.readthedocs.io/en/latest/jax-101/index.html).

# %% [markdown] id="ZRvfr609LzWu"
# ## 4. Pre-Training the model
#
# Now we will see how the power of Google's tensor processing unit (TPU) can be leveraged with Flax/JAX for the compute-intensive pre-training of language models.
#

# %% [markdown] id="_MGleTRG6Vor"
# At first, we define all relevant hyper-parameters for pretraining in this notebook:
#
# - Each TPU will process a batch size of `16`
# - The model is trained for `10` epochs
# - The learning rate starts at `3e-4` and is successfully linearly decayed with each training step
# - To reproduce the training run, a random seed is set to `0`.
#
# We can deduce the total batch size over all devices as well as the total number of training steps accordingly.

# %% [markdown] id="FB9bRDBq5j3r"
# In the [official GPT2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) a batch size of 512 is used.
#
# Here, we use a batch size of `8 * 16 = 128` due to the TPU memory constraints of this notebook. When running this script locally on a TPUv3-8, one can easily use batch sizes of up to `8 * 64 = 512`.

# %% [markdown] id="i0Tylp115u1r"
# Now we randomly initialized a `distilgpt2` model according to its configuration. To save memory and improve speed, we initialize the weights directly in `bfloat16` by setting `dtype=jnp.dtype("bfloat16")`.

# %% id="aVr9TCzfacLN"
from transformers import FlaxAutoModelForCausalLM

model = FlaxAutoModelForCausalLM.from_config(
    config, seed=training_seed, dtype=jnp.dtype("bfloat16")
)

# %% [markdown] id="sMS_QkT76Lgk"
# Next, we define the learning rate schedule. A simple and effective learning rate schedule is the linear decay with warmup (click [here](https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.get_linear_schedule_with_warmup) for more information). For simplicity, we set the number of warmup steps simply to 0 here. The schedule is then fully defined by the number of training steps and the learning rate.
#
# It is recommended to use the [**optax**](https://github.com/deepmind/optax) library for training utilities, *e.g.* learning rate schedules and optimizers.
#
# To see how to define a learning rate schedule with warmup, please take a look at the [official Flax CLM pre-training script](https://github.com/huggingface/transformers/blob/master/examples/flax/language-modeling/run_clm_flax.py).

# %% id="kfBkuV1ck4rq"
linear_decay_lr_schedule_fn = optax.linear_schedule(
    init_value=learning_rate, end_value=0, transition_steps=num_train_steps
)

# %% [markdown] id="2p0yNxeU79F2"
# We will be using the standard Adam optimizer with weight decay, called AdamW (Adam + weight decay).
#
# AdamW can easily be imported from [optax](https://github.com/deepmind/optax) and is created from the just defined learning rate schedule as well as a couple of other hyper-parameters (*beta1*, *beta2*, *epsilon*) that are hard-coded in this notebook.
#
# For more information on AdamW (Adam + weight decay), one can take a look at [this](https://www.fast.ai/2018/07/02/adam-weight-decay/) blog post.

# %% id="xRtpv_iamZd2"
adamw = optax.adamw(
    learning_rate=linear_decay_lr_schedule_fn,
    b1=0.9,
    b2=0.98,
    eps=1e-8,
    weight_decay=0.01,
)

# %% [markdown] id="6g_fEbV-72Hc"
# Next, we will create the *training state* that includes the optimizer, the loss function, and is responsible for updating the model's parameters during training.
#
# Most JAX transformations (notably [jax.jit](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html)) require functions that are transformed to have no side effects. This is because any such side-effects will only be executed once when the Python version of the function is run during compilation (see [Stateful Computations in JAX](https://jax.readthedocs.io/en/latest/jax-101/07-state.html)). As a consequence, Flax models (which can be transformed by JAX transformations) are **immutable**, and the state of the model (i.e., its weight parameters) is stored *outside* of the model instance.
#
# Models are initialized and updated in a purely functional way: you pass the state to the model when calling it, and the model returns the new (possibly modified) state, leaving the model instance itself unchanged.
#
# Flax provides a convenience class [`flax.training.train_state.TrainState`](https://github.com/google/flax/blob/9da95cdd12591f42d2cd4c17089861bff7e43cc5/flax/training/train_state.py#L22), which stores things such as the model parameters, the loss function, the optimizer, and exposes an `apply_gradients` function to update the model's weight parameters.
#
# Alright, let's begin by defining our *training state* class. We create a `TrainState` class that stores the model's forward pass as the `apply_fn`, the `params`, and the AdamW optimizer.

# %% [markdown] id="Ermxx_REtthy"
#

# %% id="JHYfR67AoKRc"
state = train_state.TrainState.create(
    apply_fn=model.__call__, params=model.params, tx=adamw
)


# %% [markdown] id="xiYCejDd81TX"
# Next, let's implement a data loader for both training and evaluation.
# The data loader can be defined as a [Python generator](https://wiki.python.org/moin/Generators) that returns a batch model input every time it is called.
#
# First, a random permutation of the whole dataset is defined.
# Then, every time the training data collator is called the next batch of the randomized dataset is extracted, converted to a JAX array and sharded over all local TPU devices.

# %% [markdown] id="L7uoTXDLUzb-"
# At each training epoch, the dataset should be shuffled and superfluous samples that make the dataset not evenly divisible by the batch size are thrown away. Instead of passing the dataset, we prepare the indices of data samples to be used for both each training epoch.
# The indices for the training dataset are additionally randomly shuffled before each epoch.

# %% [markdown] id="MU6idLb29xYu"
# During fine-tuning, we want to update the model parameters and evaluate the performance after each epoch.
#
# Let's write the functions `train_step` and `eval_step` accordingly. During training the weight parameters should be updated as follows:
#
# 1. Define a loss function `loss_function` that first runs a forward pass of the model given data input. Remember that Flax models are immutable, and we explicitly pass it the state (in this case the model parameters and the RNG). `loss_function` returns a scalar loss (using the previously defined `state.loss_function`) between the model output and input targets.
# 2. Differentiate this loss function using [`jax.value_and_grad`](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#evaluate-a-function-and-its-gradient-using-value-and-grad). This is a JAX transformation called [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation), which computes the gradient of `loss_function` given the input to the function (i.e., the parameters of the model), and returns the value and the gradient in a pair `(loss, gradients)`.
# 3. Compute the mean gradient over all devices using the collective operation [lax.pmean](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.pmean.html). As we will see below, each device runs `train_step` on a different batch of data, but by taking the mean here we ensure the model parameters are the same on all devices.
# 4. Use `state.apply_gradients`, which applies the gradients to the weights.
#
# Below, you can see how each of the described steps above is put into practice.
#
# Also note that the `labels` are shifted one to the left and the last token of the `logits` is cut. This way, the model learns to predict the **next** token as defined in causal language modeling.

# %%
def metrics_mask(labels: jnp.ndarray):
    return jnp.where(labels == 0, jnp.nan, 1)


# %% id="GjKzb0zJd-aH"
def prepare_for_model(batch):
    return {k: v for k, v in batch.items() if k in ["attention_mask", "input_ids"]}


def train_step(state, batch, dropout_rng):
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_fn(params):
        labels = batch.pop("labels")
        logits = state.apply_fn(
            **prepare_for_model(batch),
            params=params,
            dropout_rng=dropout_rng,
            train=True,
        )[0]

        logits = logits[..., :-1, :]
        onehots = onehot(labels[..., 1:], logits.shape[-1])

        loss = optax.softmax_cross_entropy(logits, onehots).mean()
        probs = jax.nn.softmax(logits, axis=-1)
        accuracy = (probs * onehots).sum(-1).mean()
        return loss, dict(accuracy=accuracy)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grad = grad_fn(state.params)
    grad = jax.lax.pmean(grad, "batch")
    new_state = state.apply_gradients(grads=grad)

    metrics = jax.lax.pmean(
        {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step), **aux},
        axis_name="batch",
    )

    return new_state, metrics, new_dropout_rng


# %% [markdown] id="nCPedI-B-FMQ"
# Now, we want to do parallelized training over all TPU devices. To do so, we use [`jax.pmap`](https://jax.readthedocs.io/en/latest/jax.html?highlight=pmap#parallelization-pmap). This will compile the function once and run the same program on each device (it is an [SPMD program](https://en.wikipedia.org/wiki/SPMD)). When calling this pmapped function, all inputs (`"state"`, `"batch"`, `"dropout_rng"`) should be replicated for all devices, which means that the first axis of each argument is used to map over all TPU devices.

# %% id="w3k1Lqerpw5k"
parallel_train_step = jax.pmap(train_step, "batch")


# %% [markdown] id="0DWFAZM6A8uf"
# Similarly, we can now define the evaluation step. Here, the function is much easier as we don't need to compute any gradients. To better monitor the performance improvement during training, the next token loss is computed and stored in a `metric` dictionary during evaluation.

# %% id="EGEv7dyfpW4p"
def eval_step(params, batch):
    labels = batch.pop("labels")

    logits = model(**prepare_for_model(batch), params=params, train=False)[0]

    logits = logits[..., :-1, :]
    onehots = onehot(labels[..., 1:], logits.shape[-1])

    loss = optax.softmax_cross_entropy(logits, onehots).mean()
    probs = jax.nn.softmax(logits, axis=-1)
    accuracy = (probs * onehots).sum(-1).mean()

    loss = optax.softmax_cross_entropy(logits, onehots).mean()

    # summarize metrics
    metrics = {"loss": loss, "accuracy": accuracy}
    metrics = dict(
        **jax.lax.pmean(metrics, axis_name="batch"), probs=probs, labels=labels
    )
    return metrics


# %% [markdown] id="guaYWTvFA_66"
# Similarly, we also apply `jax.pmap` to the evaluation step.

# %% id="0B8U2r2RpzjV"
parallel_eval_step = jax.pmap(eval_step, "batch")

# %% [markdown] id="DLaM60PCY8Ka"
# Next, we replicate/copy the weight parameters on each device, so that we can pass them to our parallelized mapped functions.

# %% id="kncZTfALp3PG"
state = flax.jax_utils.replicate(state)

# %% [markdown] id="i2xg8oI-ZJ3P"
# We can almost start training! In a final preparation step, we generate a seeded [**PRNGKey**](https://jax.readthedocs.io/en/latest/_autosummary/jax.random.PRNGKey.html#jax-random-prngkey) used as the random seed for dropout layers and dataset shuffling.
#
# Similar to how we had to copy/replicate the state on all 8 TPU devices, we also need to generate one `PRNGKey` per device, which is why we split the initial `rng` key into 8 random seeds.

# %% [markdown] id="bKuMWHicbede"
# Now, we are all set to finally start training!
# Let's put all the pieces together and write the training loop.
#
# We start each epoch by generating a new random seed that will be used for dataset shuffling, the dropout layers and the input token masking.
#
# Next, we generate the training dataset indices.
# In the first nested loop - the training loop - we shard the input batch on all 8 TPU devices, and run the training step.
#
# Analogs, in the second nested loop - the evaluation loop - the evaluation batches are sharded and the evaluation step is run.
#
# **Note**: It might seem that the following cell "hangs" when executed for the first time. This is because JAX first traces & compiles the code, the very first time it is run. After the first training step, you should notice that execution is much faster.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["0cd605a679c64f8f8c00cd0df010645e", "9b55c8d4ff3e471d9f148ae7f95690cf", "6c3d2a56f07d4bb28a44f0d6550492e2", "c5f28a576f944807aa1b6c72b006337b", "821bc913451b4c868e33989745a717a0", "913feddb35ef4a078c02186eca41b6e7", "07d1db7e0042496c972cb67bc613344d", "4a415cc59dcf4d5b9fe2cede86b31762", "3ad34c75d4484d9480be382debc8cc23", "f4550e133c704df6b6026acd45c0517b", "79a17623d6044f6e8151f53da9f32c6d", "7e4e019ca40c444c88c5a4687d395424", "87a1066ef33342579b132d0dc016bed5", "a1eaa409f58442cd9e0a4356796f6f9e", "956f235b00744defbb20cb9adddd84bb", "934e9173f1e849c3a1fa7cb8c679f18e", "d2abfc39c60e43d094b6d8d8dc225981", "7300f0ba9bac4861b9575caaddae2ed4", "28cb7fc3270c4a3f8584ab86ede9443b", "6c453a317fab4829ac5773ffebc8a2c1", "249f3b2e6c8749b2a100a04505e03b80", "027d3bb55d2d4b95a1f821be28c40366", "769097fb569f4a769644be5ff7ee8aba", "8176c3566d144f018d65f8a1a58e82d8", "4f47890a947546feb7271a2818d403de", "34d7ca70def34a9f9d45723023f1af43", "2243f28dd6d5402ab55a9bbe259daba1", "84984645cd9b42e989c247cfcfd94fab", "7549716405684e92bbeaf4a65b5bc2d9", "4f5ca2f2bf5641e895e303d8b8ed8eb4", "883cc6790d8f4447a9a1fb91c74b5169", "74013c09ed62489fb3de9cc0560304c8", "1bd45ae4736b41c9808aff64c05156a4", "1495d142ae3249319bddcfcbf7b10095", "64118bfb4e2b4e5dbc33b89e009402eb", "4643823e6c3c4d328eed6c7667ce0cca", "3ccf5451fea24ec5ba8b50c0b3bdc79b", "3e706537172048d999ba86bd917fb2c7", "af911e7105dd479d9f49ee9a15e99517", "9e9b875fd8d34647b7e532e48b354bdc", "2b18ca7d5ca84173a8ce41a0d11f1a2a", "3d79bdf68d2341ffa39248bbd3ba198e", "0b1c9f18d3ac461a8f34838635f3ff33", "53ef4f74c0a14e46a39e5ffab354033c", "f21a54f827d94ff299d20f4a20c15ca3", "df945afe6ffe4d4cae6dd8e09f2ead97", "86e0d552fc3f4627a214ed520902c24f", "52b89535b0a1461db5e236041a4f96bd", "a15cb9ecfbeb497bb6dfeba9cd7f66c9", "109e4d26d86f47bdbadfc6e55e6d8972", "2afaa1cd03fa4902885e343aa990b25b", "0256d3b5c72f4c3b8d7cbe5f599bdf3c", "d632fdc6e08c42cdb92afb57c5ae9268", "bec398b9c204408da2cc755a3685a4c0", "76920d30bb8a41d3ab2b5c3117cf0fb4", "72067ef2796f4efea8bdaab791c7de99", "e55ca4a9a89e4ffbbc9b9a646872bac9", "897f0ff47f494305bc51d98ba2daca54", "53e77226c17745599a76b70ec94e0b34", "76d12b31035d4bcda5d47218c5e5c9a5", "fedf7de0fa424d90addf0df69b8f86d4", "7eac0e2868ec41e697fa678ad750c578", "6292133a08e445f0a6f32dce0b4ec49b", "d04690a7c577440ba01becf444da769a", "2df61da968434cc9aad7868ab06bcc45", "44a672cfbc364f1da15afb7a8334a1a1", "b59ac8ef838a446ea7e4bc264de42b1e", "b7b47dd3c5fb4f729346f24cfbf0e0aa", "1d821267a8294a87a6d182eb92708e0f", "6dee721f063249ce99f7503918852a8f", "d7d5878371074cd0b66322e548896b8b", "fb4c81abc339458c920063034fd96e58", "bdf5084b179e497ca76bc08d977bf31f", "c6bcb8fbe4b746d39f94abd14c9a5462", "d194fe11f4c848a9b2c47c948e478e63", "bd6db04835824eb585f70e728ad729be", "b26ec547a0c54ffabd1daa5fa289be1b", "be6655a59cd147b4b72cc1953252637c", "84326265524d4ab59f47160c902fb486", "daab1b63c723454c92436d03257d7c01", "fb881aa15c9b4a56ac05c02ef0a1df2b", "51bfa23b0c304f6abb71576ebf4f293b", "231b1d54cce34263ab9374989d2bc36e", "474702c024c040b98e5b4d2637870cb1", "94450fdd44ce46c58b07aac2265335a3", "ae1321fd38434ec6a9846d0e6fc2cac7", "9b1536f754c84ab19a6eedca9760c64c", "786f957ab25c441e8041599948e3974f", "9b51318174b14f06acc44fd1b78ab017", "ad557fc4f77a4ac1ade64cc8558a1d06", "4ff19b18ac364f76b0fdee4958a4046c", "139ed3aeecec4f16ada32163be201095", "b9baa2f5812248f281ae944498a32e85", "e78d08b21ec3470897f9b557b0e95435", "ec21d88f1eb04f2b8b46e9629b54e260", "11881b386f574afc8820aafd6aaf782a", "901015c6810e4c43b234c615fab97c91", "e0a81ad163bf4ee99d8388dba81f2103", "a1bd8f00663f4395b0ce886e3174f25d", "8b7c2a819e384b7faede144a470df052", "88697e56e296453cb3e9102bbe45d036", "106c0a5554b045fda49b0efdbf804618", "8a394eb3cad44f7587167442cab457d4", "421e335b8eae41578156f0e462457005", "1f20d13e30504f0b91a0718d5518ccbd", "156c20a0357f4a6cbe517c29aa80b2cf", "9d967ed9c89b408db984e41793c6765b", "5494324d77ec4c5d8454cde7514f1650", "9f7bd2b8779a4944bfdb9c4284a6c33e", "7246a61f099c4b00acfc15f2bcc60eb6", "dec75e54353e488a912fea7e3b0bcf81", "9d97171fd3b14f18b6dc259cab8104ee", "5a4b1877d8254f1ea8a3c8390b1deb0c", "1a0159ffb55942d0a0e414a135e15b23", "fc4d5ef1b03f44978fa65df3e5f823a6", "d3339f69e7b44f0d993fe6774a69cf18", "db81a51285b348e7918da055485fe0e8", "507c0e9fe85748e1bf5803fada529153", "aa3ae46342764f5fb312ac668147b17c", "d6db384354c747fa979e245292ed6d54", "f8b992cfff6a457db52f831311ec402b", "de7dbe962f524b8abee756b3f0e331eb", "8b9336be2afa4e94aa888e71a210c091", "b8112086cbf64757b4c86ce82f41b7ba", "5b46c12296634b7a84361b41246c4324", "889c57acee8848c6bc534c45663f5b1d", "4bfdd6289d4d48b1b7deb4adfb194c53", "c4824da867004c84bf3446fa18bccc62", "677db8d6281c447ebd6ff81d1ac8dc96", "e1e2e1f7357247b2b8e36eb69d059322", "3808d154f02346c28e0569d4400a960c", "f200800f3e4441c79b08132d7f566da9", "bc0f2ef41b984e898e194f46b96df56c", "4f5d9a8d40304e359da9b860d39156e0", "e96b78d455784610b01b74610af03e56", "aa2151f469254d73add7475bb725846b", "528ae213a037478f82c7283becebc745", "67d367acde1249cb8950315c9e9e3133", "5b9f27f9f924473d8797d17e75fc2fb0", "ecd1dfd1153a4f32b75e4e3836dcce27", "c0eb38c8a84840e9aa1ba918135663b9", "830a673ed8cd427faf9cdfa8fa4f558c", "0760b51d75f44ce3bf2d70bcce322350", "3fd185f46aeb46d28c91ad9742f52a0a", "63bbf58e16504b859f1bd7d329ed2cff", "58c2a739adf24b47a846568f94412020", "d5ced5bd5146470ca2caf80aadac8e40", "6be03ff88ee44da3a7dfebab08774b91", "63d87af913154d30a9955c1e3af62729", "c34a1f7e04e546069ad479ec7a2c5b2b", "824d9e508e7045a2a358a7af3bae926d", "165981935e5f46af95df3d2f5bf56bc0", "4b861a6c6efa4960824e1dfb3488738f", "00568e24a96747979facc0773bdc00b5", "0cef18d68f73417981e9950069580b00", "3c6d2532d5cf47a2b0cfaa33865800f2", "00034ee778b7494ebe2c2a93fd5e3dc7", "6e5e572639c54ec0b10ed6a5d86f5511", "a38f0fc8770c45869a594c858b9f0b21", "4f5d6a93d7c040149e1b02b3897a938f", "88b29d7427d0471ba7bc82233457aa0b", "cf9f76a4fb214e38b418d6900d05f569", "e9e770030ef049be801bcc7283886157", "4d59faff8f4a436ba93b402c308f0f15", "8797e74bdc79441aada81fd833534480", "db71c278146a47f983c3752c51bde34b", "715ec81541534791a1b22dcbb11ae32e", "610388e1b63e4142a247b35b99aa0dfe", "dd7da90940894a4f97cb69b92db023dd", "28c810c2c2e344dda926604c65fb3eee", "f52aa5f89f8940a1a1f7dec4dc9b71c5", "2c67c37661c7486b8b4e0727560b4f4f", "8395b5c538514ed1b50f106dba1f71f4", "b3bef88b631941e1affe0060e5cfee7b", "55258666ca7645079c6c39374444179d", "037f2f241c71401a8c913dbaafa2c6a0", "121c991fc9294cb58607edf5f6d2632f", "8b5139447eb541cb8cb0d44be1617350", "a72b77b951194099a945a526a5136c08", "cab90c373d914c2ca1778213b2ca5fd6", "132609282a46484da3b44bcdba4766f6", "e367afc356784c33a0c5bad3b6389f0b", "eac81b16b6754261af0f7c0615a387f5", "2aeda2b9b7bc4a1f80bfc303b5a383da", "071a8a77c1bb4165b7970f61bee80a1c", "f2360a80edc144b3b2a623b200d809fe", "e0eaffb1c4f348989f6b78eab0f6af78", "b7bc17cf8bf94aa1a86c90a18c28386f", "ffb0bd2b68ee4304a7695de0b79d346a", "a94b72d93e8a45489604bbbffbafd371", "36f7b5e84ec54a9f8af542df315cbda7", "0556e199246c4aaabdf91fa4c44c3971", "e5ef3597825d4b1e914f9e0147fe3765", "de7dc0e65af84b40852e1a2e6c996643", "48cd49e3a99745afa342ab21b1694f29", "c8dd2711117347f6a27be5e79ac3939a", "0e29b4d0f99f4518b0d84a6d693e3997", "afba5ae9e3fe449996ec3e4ba05b97da"]} id="U946A-YZp-Pe" outputId="da45081c-2c7e-472f-b1f2-fa2489479549"
import seaborn as sns
import matplotlib.pyplot as plt

losses = dict(train=[], validation=[])
accuracies = dict(train=[], validation=[])

for epoch in tqdm(range(1, num_epochs + 1), desc=f"Epoch ...", position=0, leave=True):
    rng, input_rng = jax.random.split(rng)

    # -- Train --
    train_loader = data_loader(
        input_rng, my_datasets["train"], total_batch_size, shuffle=True
    )
    with tqdm(
        total=len(my_datasets["train"]) // total_batch_size,
        desc="Training...",
        leave=False,
    ) as progress_bar_train:
        for model_inputs in train_loader:
            # Model forward
            state, train_metric, dropout_rngs = parallel_train_step(
                state, model_inputs, dropout_rngs
            )

            progress_bar_train.update(1)

        loss = train_metric["loss"].mean()
        losses["train"].append(loss)
        accuracy = train_metric["accuracy"].mean()
        accuracies["train"].append(accuracy)
        progress_bar_train.write(
            f"Train... ({epoch}/{num_epochs} | Accuracy: {round(accuracy, 3)} | Loss: {round(loss, 3)}, Learning Rate: {round(train_metric['learning_rate'].mean(), 6)})"
        )

    # -- Eval --
    eval_loader = data_loader(input_rng, my_datasets["validation"], total_batch_size)
    eval_metrics = []

    with tqdm(
        total=len(my_datasets["validation"]) // total_batch_size,
        desc="Evaluation...",
        leave=False,
    ) as progress_bar_eval:
        for model_inputs in eval_loader:
            # Model forward
            eval_metric = parallel_eval_step(state.params, model_inputs)
            eval_metrics.append(eval_metric)

            progress_bar_eval.update(1)

        probs = [m["probs"] for m in eval_metrics]
        labels = [m["labels"] for m in eval_metrics]
        eval_metrics = get_metrics(eval_metrics)
        eval_metrics = jax.tree_map(jnp.mean, eval_metrics)

        loss = eval_metric["loss"].mean()
        losses["validation"].append(loss)
        accuracy = eval_metric["accuracy"].mean()
        accuracies["validation"].append(accuracy)

        progress_bar_eval.write(
            f"Eval... ({epoch}/{num_epochs} | Accuracy: {round(accuracy, 3)} | Loss: {round(loss, 3)})"
        )

    for label, loss in losses.items():
        plt.plot(loss, label=label)
    plt.yscale("log")
    plt.show()
    for label, accuracy in accuracies.items():
        plt.plot(accuracy, label=label)
    plt.show()


# %% [markdown] id="ZI4XIhY-7hyh"
# It can be seen that in this colab training already reaches a speed of 2.42 training steps per second. Executing [**`run_clm_flax.py`**](https://github.com/huggingface/transformers/tree/master/examples/flax/language-modeling/run_clm_flax.py) on a TPUv3-8 VM should be as fast as 7 training steps per second.
#
# For a more in-detail comparison of runtimes please refer to [this](https://github.com/huggingface/transformers/tree/master/examples/flax/language-modeling#runtime-evaluation) table.

# %% id="ocu-dPKwW8kJ"
del model
del eval_step
del train_step
del state
