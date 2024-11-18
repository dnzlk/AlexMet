![Metal logo](https://github.com/dnzlk/AlexMet/blob/main/metal.jpg)

**Disclaimer**\
This is not an ultimate deep learning framework for any possible task. It is an implementation of a certain architecture so it was implemented with a certain architecture in mind.

Hi.\
If you are reading this then you are probably a nerd. At least I hope so.\
I've decided to implement **AlexNet** in "pure" Metal. Boys ask their moms to buy them an Nvidia GPU. Men use what they have until it burns down.\
I beleive that today's Mac computers with their fancy M chips are good enough for training at least "simple" neural nets like this one.

All codebase is written in Objective-C, Metal & C.\
It runs pretty good on my machine, but it is still not good enough.

### Problems

**1.0** It is really slow in a Convolutional layer backward pass because of atomics. I suspect that it can be solved using threads synchronization.\
**2.0** Forward pass takes about 2 seconds for 128 elements batch on my M1 Pro and I beleive that it must also be faster.

If you know how to solve those problems please create PRs or text me on [X](https://x.com/ohuyba).

