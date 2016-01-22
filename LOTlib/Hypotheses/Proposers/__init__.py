

class ProposalFailedException(Exception):
    """
    This gets raised when we have a proposal that can't succeed
    """
    pass

from Regeneration import regeneration_proposal
from InsertDelete import insert_delete_proposal
