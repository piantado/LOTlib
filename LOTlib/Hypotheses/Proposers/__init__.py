class ProposalFailedException(Exception):
    """
    This gets raised when we have a proposal that can't succeed
    """
    pass

from CopyRegenProposal import copy_regen_proposal
from InsertDeleteProposal import insert_delete_proposal
from RegenerationProposal import regeneration_proposal
from CopyRegenProposal import copy_proposal
