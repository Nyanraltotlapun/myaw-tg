#!/usr/bin/python3

# Myaw-TG is Telegram chat history analyzer
#     Copyright (C) 2024  Kirill Harmatulla Shakirov
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

import configparser
import sys
import os
import math
import json
import argparse
from datetime import datetime
import itertools

C_OTHER = '\033[93m'
C_INFO = '\033[32m'
C_WARNING = '\033[34m'
C_ERROR = '\033[01m' + '\033[31m'
C_DEBUG = '\033[36m'
C_RESET = '\033[0m'

def log_error(message):
    log_text = f"{C_ERROR}{datetime.now().isoformat()} ERROR: {message}\n{C_RESET}"
    sys.stdout.write(log_text)

def log_warning(message):
    log_text = f"{C_WARNING}{datetime.now().isoformat()} WARNING: {message}\n{C_RESET}"
    sys.stdout.write(log_text)

def log_info(message):
    log_text = f"{C_INFO}{datetime.now().isoformat()} INFO: {message}\n{C_RESET}"
    sys.stdout.write(log_text)

def load_config(config_file_name: str):
    if not os.path.exists(config_file_name):
        raise FileNotFoundError(f"Config file ({config_file_name}) not found!")

    return configparser.ConfigParser().read(config_file_name)

class UserStats:
    def __init__(self, from_id: str, name: str, chars_chain_len = 2, words_chain_len = 2):
        self.words_chain_len = words_chain_len
        self.chars_chain_len = chars_chain_len
        self.messages_count: int = 0
        self.chars_markov_chain = MarkChain()
        self.words_markov_chain = MarkChain(start_token="#START#", end_token="#END#")
        self.chars_entropy: float = 0.0
        self.words_entropy: float = 0.0
        self.mean_entropy: float = 0.0
        self.chars_per_message: float = 0.0
        self.words_per_message: float = 0.0

        self.from_id = from_id
        self.id = from_id[4:]
        self.name = name

    def process_message(self, msg: dict):
        self.messages_count += 1

        for ent in msg["text_entities"]:

            if not filter_entry(ent):
                continue

            text = ent["text"]

            # count chars
            cur_token = self.chars_markov_chain.start_token
            for cur_char in text:
                self.chars_per_message += 1.0
                self.chars_markov_chain.count_link(cur_token, cur_char)
                cur_token = cur_char
            self.chars_markov_chain.count_link(cur_token, self.chars_markov_chain.end_token)

            # count words
            words = text.split()
            cur_token = self.words_markov_chain.start_token
            for cur_word in words:
                self.words_per_message += 1.0
                self.words_markov_chain.count_link(cur_token, cur_word)
                cur_token = cur_word
            self.words_markov_chain.count_link(cur_token, self.words_markov_chain.end_token)

    def calc_stats(self):
        self.chars_entropy = calc_mark_entropy(self.chars_markov_chain, self.chars_chain_len)
        self.words_entropy = calc_mark_entropy(self.words_markov_chain, self.words_chain_len)
        self.mean_entropy = (self.chars_entropy + self.words_entropy) * 0.5
        self.words_per_message = self.words_per_message / self.messages_count
        self.chars_per_message = self.chars_per_message / self.messages_count

    def to_string(self) -> str:
        stats_list = [str(self.id), f"\"{self.name}\"", f"\"{self.from_id}\"", str(self.messages_count),
                      f"{self.mean_entropy:.5f}", f"{self.chars_entropy:.5f}", f"{self.words_entropy:.5f}",
                      f"{self.chars_per_message:.2f}", f"{self.words_per_message:.2f}"]

        return ",".join(stats_list)

    def __str__(self):
        return self.to_string()

class MarkChain:
    def __init__(self, start_token: str = "SS", end_token:str = "EE"):
        self.start_token = start_token
        self.end_token = end_token
        self.chain = dict()
        self._chain_prob = None

    def count_link(self, c1: str, c2: str):
        """

        :param c1:
        :type c1:
        :param c2:
        :type c2:
        """
        if self.chain.get(c1) is None:
            self.chain[c1] = {c2: 1}
        else:
            self.chain[c1][c2] = self.chain[c1].get(c2,0) + 1

    def chain_prob(self) -> dict:
        if self._chain_prob is None:
            return self.update_chain_prob()
        return self._chain_prob

    def update_chain_prob(self) -> dict:
        res = dict()
        for c1, counts in self.chain.items():
            probs = dict()
            total = sum(counts.values())
            for c2, toke_count in counts.items():
                probs[c2] = float(toke_count)/float(total)
            res[c1] = probs
        self._chain_prob = res
        return res

    def state_probability(self, state_seq) -> tuple[float, float, any]:
        """
        Calculate probability of state specified by tokens sequence.
        :param state_seq: Sequence representing chain state.
        :type state_seq: Iterable
        :return: Tuple of two floats and one token.
        First value is probability of state, second value is probability of last state chain,
        And third value is last token in the chain.
        :rtype: tuple[float, float, any]
        """
        seq_iter = state_seq.__iter__()
        seq_prob: float = 1.0
        cur_prob: float = 0.0
        cur_token = next(seq_iter)
        for next_token in seq_iter:
            cur_prob = self._chain_prob[cur_token].get(next_token, 0.0)
            if cur_prob == 0.0:
                return 0.0, 0.0, next_token
            seq_prob *= cur_prob
            cur_token = next_token

        return seq_prob,cur_prob,cur_token


def filter_entry(ent: dict)-> bool:
    if ent["type"] in ["plain"]:
        return True
    return False



def calc_mark_entropy(chain:MarkChain, chain_len: int) -> float:
    if len(chain.chain) == 0:
        return 0.0

    H = 0.0
    chain.update_chain_prob()
    possible_tokens = set(chain.chain_prob().keys())
    #possible_tokens.add(chain.end_token)
    possible_tokens.remove(chain.start_token)
    for tokens_seq in itertools.product(possible_tokens, repeat=(chain_len-1)):
        # Pq probability of sequence
        tokens_seq_list = [chain.start_token]
        tokens_seq_list.extend(tokens_seq)
        #log_info(f"Processing sequence: {tokens_seq_list}")
        Pq,_,_token = chain.state_probability(tokens_seq_list)
        if Pq != 0.0:
            for last_token in possible_tokens:
                last_t_prob,_,_ = chain.state_probability((_token, last_token))
                if last_t_prob != 0.0:
                    H += (Pq* last_t_prob) * math.log2(last_t_prob)
            #process END Token separately
            last_t_prob, _, _ = chain.state_probability((_token, chain.end_token))
            if last_t_prob != 0.0:
                H += (Pq * last_t_prob) * math.log2(last_t_prob)

    return H * -1.0


def main():
    # Initialize arguments parser
    parser = argparse.ArgumentParser(
        prog="myaw-tg.py",
        description="This program analyze TG chat history exported in json format and calculates users messages entropy",
        epilog="Have a nice day!")

    parser.add_argument("-i", "--input-file",
                        action="store",
                        default=None,
                        help="exported history file name, for example results.json",
                        required=True)
    parser.add_argument("-o", "--output-file",
                        action="store",
                        default=None,
                        help="Statistic output file name, for example stats.csv",
                        required=True)


    arguments = parser.parse_args()

    log_info(f"Reading data from file: {arguments.input_file}")

    try:
        with open(arguments.input_file, "rt") as in_file:
            in_json = json.load(in_file)
    except FileNotFoundError:
        log_error(f"Cannot find chat history file: {arguments.input_file}.")
        log_info("Exiting.")
        exit(1)

    log_info(f"Total {len(in_json["messages"])} messages.")
    log_info(f"Start processing...")
    users_stats: dict = {}
    for msg in in_json["messages"]:
        if msg["type"] == "message":
            from_id = msg["from_id"]
            u_stats = users_stats.get(from_id)
            if u_stats is None:
                u_stats = UserStats(from_id, msg["from"])
                users_stats[from_id] = u_stats
            # process message
            u_stats.process_message(msg)

    log_info("Calculating users messages entropy...")
    for from_id,stats in users_stats.items():
        log_info(f"Processing user: {stats.name}...")
        stats.calc_stats()
    log_info("Done!")

    log_info("Preparing and writing stats to CSV file...")
    stats_list = list(users_stats.values())
    stats_list.sort(key=lambda x: x.messages_count, reverse=True)
    columns_names = ["id", "name", "raw_id", "messages_count",
                      "mean_entropy", "chars_entropy", "words_entropy",
                      "chars_per_message", "words_per_message"]
    columns_names_str = ",".join(map(lambda x: f"\"{x}\"", columns_names))
    with open(arguments.output_file, "wt") as out_file:
        out_file.write(columns_names_str)
        out_file.write("\n")
        for u_st in stats_list:
            out_file.write(u_st.to_string())
            out_file.write("\n")

    log_info("Done!")
    log_info("Exiting...")


if __name__ == '__main__':
    main()


