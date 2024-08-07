import random
import math
import scipy.stats as stats
from typing import List, Tuple, Dict, Optional

class Node:
    def __init__(self, id: int, dist=stats.uniform) -> None:
        self.id = id
        self.energy = 0
        self.dist = dist
        self.has_message = False

    def harvest_energy(self) -> None:
        self.energy = self.dist.rvs()

    def send_data(self, channel_gain: float) -> bool:
        if not self.has_message:
            return False
        success = random.uniform(0, 1) < self.probability_of_success(channel_gain)
        self.has_message = not success
        return success

    def probability_of_success(self, channel_gain: float) -> float:
        return min(1, self.energy * math.exp(-channel_gain))

    def __str__(self) -> str:
        return f"Node(id={self.id}, energy={self.energy:.2f})"

    def __repr__(self) -> str:
        return self.__str__()

class Channel:
    def __init__(self, id: int, nodes: List[Node], dist=stats.expon, per_user=True) -> None:
        self.id = id
        self.nodes = nodes
        self.dist = dist
        self.per_user = per_user
        self.gains: Dict[Node, float] = self.generate_gains()

    def generate_gains(self) -> Dict[Node, float]:
        if self.per_user:
            gains = self.dist.rvs(size=len(self.nodes))
            return {node: gains[i] for i, node in enumerate(self.nodes)}
        gains = self.dist.rvs()
        return {node: gains for node in self.nodes}

    def __str__(self) -> str:
        return f"Channel(id={self.id}, gains={self.gains})"

    def __repr__(self) -> str:
        return self.__str__()

class Network:
    def __init__(self, num_nodes=10, per_user=True, match_before=False, keep_alive=True):
        self.num_nodes = num_nodes
        self.per_user  = per_user
        self.match_before = match_before
        self.keep_alive = keep_alive
        self.reset()

    def reset(self):
        self.nodes    = [Node(i) for i in range(self.num_nodes)]
        self.channels = [Channel(i, self.nodes, per_user=self.per_user) for i in range(self.num_nodes)]

    def harvesting_slot(self):
        for channel in self.channels:
            channel.gains = channel.generate_gains()
        energies = self.nodes[0].dist.rvs(size=self.num_nodes)
        for i, node in enumerate(self.nodes):
            node.energy = energies[i]
            
    def sending_slot(self, protocol) -> int:
        if self.match_before:
            return protocol.execute(self.nodes.copy(), self.channels.copy())
        else:
            return protocol.execute(self.sending_nodes().copy(), self.channels.copy())
    
    def create_messages(self, input_rate):
        for node in self.nodes:
            generated = random.uniform(0, 1) < (input_rate / self.num_nodes)
            if self.keep_alive:
                node.has_message = node.has_message or generated
            else:
                node.has_message = generated
            
    def sending_nodes(self) -> List[Node]:
        return [node for node in self.nodes if node.has_message]
    
    def simulate_rate(self, protocol, rate, trial_length) -> Tuple[int, int]:
        self.reset()
        total_result = 0
        for _ in range(trial_length):
            self.create_messages(rate)
            self.harvesting_slot()
            result = self.sending_slot(protocol)
            total_result += result
        return rate, total_result / trial_length
    
    def simulate(self, protocol: 'Protocol', input_rates, trial_length=200) -> Dict[int, int]:
        results = {rate: 0 for rate in input_rates}
        for rate in input_rates:
            rate, result = self.simulate_rate(protocol, rate, trial_length)
            results[rate] = result
        return results

class MultiThreadedNetwork(Network):
    def simulate(self, protocol, input_rates, trial_length=200) -> Dict[int, int]:
            results = {rate: 0 for rate in input_rates}
            from concurrent.futures import ThreadPoolExecutor
            thread_networks = {rate: Network(self.num_nodes) for rate in input_rates}
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(thread_networks[rate].simulate_rate, protocol, rate, trial_length) for rate in input_rates]
                for future in futures:
                    rate, result = future.result()
                    results[rate] = result
            return results