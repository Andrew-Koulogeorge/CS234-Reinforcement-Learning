import torch
import torch.nn as nn
import torch.distributions as ptd

from network_utils import np2torch, device


class BasePolicy:
    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: instance of a subclass of torch.distributions.Distribution

        See https://pytorch.org/docs/stable/distributions.html#distribution

        This is an abstract method and must be overridden by subclasses.
        It will return an object representing the policy's conditional
        distribution(s) given the observations. The distribution will have a
        batch shape matching that of observations, to allow for a different
        distribution for each observation in the batch.
        """
        raise NotImplementedError

    def act(self, observations, return_log_prob = False):
        """
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            sampled_actions: np.array of shape [batch size, *shape of action]
            log_probs: np.array of shape [batch size] (optionally, if return_log_prob)

        TODO:
        Call self.action_distribution to get the distribution over actions,
        then sample from that distribution. Compute the log probability of
        the sampled actions using self.action_distribution. You will have to
        convert the actions and log probabilities to a numpy array, via numpy(). 

        You may find the following documentation helpful:
        https://pytorch.org/docs/stable/distributions.html
        """
        observations = np2torch(observations) # make observations a tensors

        #######################################################
        #########   YOUR CODE HERE - 1-4 lines.    ############

        action_distro = self.action_distribution(observations) # get probability distributions for next action given each state 
        sampled_actions = action_distro.sample() # sample an action from each probability distributions
        log_probs = action_distro.log_prob(sampled_actions).detach().numpy() # compute log probability of the density of sampled action
        
        #######################################################
        #########          END YOUR CODE.          ############

        if return_log_prob:
            return sampled_actions, log_probs
        return sampled_actions


class CategoricalPolicy(BasePolicy, nn.Module):
    def __init__(self, network):
        nn.Module.__init__(self)
        self.network = network

    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: torch.distributions.Categorical where the logits
                are computed by self.network

        See https://pytorch.org/docs/stable/distributions.html#categorical
        """
        #######################################################
        #########   YOUR CODE HERE - 1-2 lines.    ############
        
        # support of categorical distribution is finite (takes on discrete values)
        distribution = ptd.Categorical(logits=self.network(observations)) # network computes logits for categorical distribution
    
        #######################################################
        #########          END YOUR CODE.          ############
        return distribution


class GaussianPolicy(BasePolicy, nn.Module):
    def __init__(self, network, action_dim):
        """
        After the basic initialization, you should create a nn.Parameter of
        shape [dim(action space)] and assign it to self.log_std.
        A reasonable initial value for log_std is 0 (corresponding to an
        initial std of 1), but you are welcome to try different values.
        """
        nn.Module.__init__(self)
        self.network = network
        #######################################################
        #########   YOUR CODE HERE - 1 line.       ############

        self.log_std = nn.Parameter(data=torch.zeros(action_dim))

        #######################################################
        #########          END YOUR CODE.          ############

    def std(self):
        """
        Returns:
            std: torch.Tensor of shape [dim(action space)]

        The return value contains the  ** standard deviations for each dimension
        of the policy's actions. ** It can be computed from self.log_std
        """
        #######################################################
        #########   YOUR CODE HERE - 1 line.       ############
        std = torch.exp(self.log_std)
        #######################################################
        #########          END YOUR CODE.          ############
        return std

    def action_distribution(self, observations):
        """
         NOTES: For Gaussian policy, the distribution we are sampling over is continuous. 


        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: an instance of a subclass of
                torch.distributions.Distribution representing a diagonal
                Gaussian distribution whose mean (loc) is computed by
                self.network and standard deviation (scale) is self.std()

        Note: PyTorch doesn't have a diagonal Gaussian built in, but you can
            fashion one out of
            (a) torch.distributions.MultivariateNormal
            or
            (b) A combination of torch.distributions.Normal
                             and torch.distributions.Independent
        """

        #######################################################
        #########   YOUR CODE HERE - 2-4 lines.    ############

        # compute the mean of the distribution from the nn
        mean = self.network(observations) # nn :(b x dim(obseration)) --> (b x dim(action)) : EXPECTED ACTION VALUE FOR STATE IN THE BATCH

        # compute the variance from the learnable parameters
        std = self.std() # vector in dim[(action space)] : std of action distribution same for all states!

        # compute the distribution of actions for each state with the mean and diag(variance)
        distribution = ptd.MultivariateNormal(loc=mean, covariance_matrix=torch.diag_embed(std.pow(2))) # mean different for each distro; std the same

        #######################################################
        #########          END YOUR CODE.          ############
        
        
        return distribution 
