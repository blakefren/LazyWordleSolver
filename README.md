# LazyWordleSolver

## Background

This [Wordle](https://www.nytimes.com/games/wordle/index.html) solver uses a lazy method to find the daily puzzle's solution. It operates under the following assumptions:

1. No solution is ever repeated.
2. The full list of words is known and constant.

[This](https://www.washingtonpost.com/video-games/2022/11/07/wordle-new-answers-new-york-times-update/) is an additional update to the rules by the NYT, including the following:

1. TODO

So, the process we'll follow is:

1. For each guess, find the best word to guess via an entropy-reduction method, which should eliminate as many words as possible.
2. Then, take the feedback from the guess and eliminate all words that don't meet the pattern.
3. From the remaining words, again find the best word via entropy reduction and guess it.
4. Continue until there is a single word remaining (and guess it), or until we run out of guesses.

Some other references:

* [Finding the optimal human strategy for Wordle using maximum correct letter probabilities and reinforcement learning](https://arxiv.org/abs/2202.00557?context=cs)
* [The mathematically optimal first guess in Wordle](https://medium.com/@tglaiel/the-mathematically-optimal-first-guess-in-wordle-cbcb03c19b0a)


## Setup

1. Install OpenAI's `gym` package via `pip install gym`.
    * Note that `gym` has recently become `gymnasium` ([link](https://www.gymlibrary.dev/)), which can be installed via `pip install gymnasium`, and may require some import/API modifications.
2. Install [this](https://github.com/DavidNKraemer/Gym-Wordle) `gym` version of Wordle (from [DavidNKraemer](https://github.com/DavidNKraemer)) via `pip install gym-wordle`, but any version should do.

Note that `gym-wordle` contains its own lists of valid words and solutions. If you're using an environment that doesn't have these lists, you can download them from sources like:
* Wordle dictionary: https://gist.github.com/cfreshman/cdcdf777450c5b5301e439061d29694c
* List of past Wordle solutions from: https://www.rockpapershotgun.com/wordle-past-answers

## How to run

TODO
