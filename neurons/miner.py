# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Omega Labs, Inc.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import typing
import bittensor as bt

# Bittensor Miner Template:
import omega

from omega.base.miner import BaseMinerNeuron
from omega.imagebind_wrapper import ImageBind
from omega.miner_utils import search_and_embed_videos
from omega.augment import LocalLLMAugment, OpenAIAugment, NoAugment
from omega.utils.config import QueryAugment
from omega.constants import VALIDATOR_TIMEOUT

import json
from omega.miner_utils import find_query_augment, insert_query_augments, check_query_augment, update_query_augment


class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        query_augment_type = QueryAugment(self.config.neuron.query_augment)
        if query_augment_type == QueryAugment.NoAugment:
            self.augment = NoAugment(device=self.config.neuron.device)
        elif query_augment_type == QueryAugment.LocalLLMAugment:
            self.augment = LocalLLMAugment(device=self.config.neuron.device)
        elif query_augment_type == QueryAugment.OpenAIAugment:
            self.augment = OpenAIAugment(device=self.config.neuron.device)
        else:
            raise ValueError("Invalid query augment")
        self.imagebind = ImageBind()
        
        self.concurrent_requests = 0

    async def forward(
        self, synapse: omega.protocol.Videos
    ) -> omega.protocol.Videos:
        bt.logging.info(f"Received scraping request: {synapse.num_videos} videos for query '{synapse.query}'")
        start = time.time()
        
        # check to see if we have an augment ready to use for this query
        query_to_use = find_query_augment(synapse.query)
        if (not query_to_use):
          query_to_use = self.augment(synapse.query)
        else:
          # if we found one in the db, increment the use count
          update_query_augment(query_to_use)
          bt.logging.info(f"Found augmented query in db: {query_to_use}")
   
        synapse.video_metadata, optimized_query = await search_and_embed_videos(
            synapse.query, query_to_use, synapse.num_videos, self.imagebind, start, self.augment
        )
        if optimized_query is not None and optimized_query != "":
          synapse.query = optimized_query
        bt.logging.info(f"Submitting optimized query: {synapse.query}")
        time_elapsed = time.time() - start
        if len(synapse.video_metadata) == synapse.num_videos and time_elapsed < VALIDATOR_TIMEOUT:
            bt.logging.info(f"–––––– SCRAPING SUCCEEDED: Scraped {len(synapse.video_metadata)}/{synapse.num_videos} videos in {time_elapsed} seconds.")
        else:
            bt.logging.error(f"–––––– SCRAPING FAILED: Scraped {len(synapse.video_metadata)}/{synapse.num_videos} videos in {time_elapsed} seconds.")
        
        # request is complete, reduce concurrency count
        #self.concurrent_requests -= 1
        return synapse

    async def blacklist(
        self, synapse: omega.protocol.Videos
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (template.protocol.Videos): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            # Ignore requests from un-registered entities.
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"
        
        if self.metagraph.S[uid] < 1000:
          bt.logging.warning(f"Ignoring request from validator/uid with less than 1000 TAO staked.")
          return True, "Not even staked TAO"
        
        # if too many concurrent requests, pass. Right now only doing 2 at a time.
        #if self.concurrent_requests >= 2:
        #  bt.logging.warning(f"Too many concurrent requests, passing on this validator request.")
        #  return True, "Too many concurrent requests, passing."
        # let's keep track of how many concurrent requests we're handling.
        #self.concurrent_requests += 1
        
        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: omega.protocol.Videos) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (template.protocol.Videos): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        prirority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", prirority
        )
        return prirority

    def save_state(self):
        """
        We define this function to avoid printing out the log message in the BaseNeuron class
        that says `save_state() not implemented`.
        """
        pass


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            #bt.logging.info("Miner running...", time.time())
            time.sleep(5)
