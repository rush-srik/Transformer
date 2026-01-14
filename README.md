# JAX Transformer from Scratch 

A pure JAX implementation of the Transformer architecture.

This project was built to understand the "engine" of Large Language Models by implementing the core architecture, optimization logic, and training loop entirely from scratch using `jax.numpy` and linear algebra, without relying on high-level neural network libraries like Flax, Haiku, or Optax.

## Key Features

* **Pure JAX Implementation:** No high-level layers. Multi-Head Attention, LayerNorm, and Feed-Forward blocks are implemented as raw matrix operations.
* **Custom Optimizer:** Manual implementation of the AdamW optimizer (Adaptive Moment Estimation with Decoupled Weight Decay), including explicit handling of moment tracking and bias correction.
* **High-Performance Training:** Fully JIT-compiled training steps using `@jax.jit` with static argument handling for maximum speed.
* **Advanced Regularization:** Implemented Dropout and Cosine Learning Rate Decay with linear warmup to stabilize convergence.
* **Experiments:** Trained to reverse sequences with 100% accuracy using an Encoder-Decoder Transformer and generate coherent pseudo-Shakespearean text with a perplexity of ~6 using a Decoder-Only Transformer.

## Sample Shakespeare Output

*Generated after 5,000 steps of training with temperature=0.8:*

>GREGORY BELIZNE:
>Poing not, nor his confent thou hast off;
>And made would reads, my think horse wall powen
>That far's report on guilss from thee
>That what man; but the quone of it barry,
>To with for for my heart of nurse by friend,
>To day offend full Place of the friend, but
>Whom me me in his eage to hours: the the deam?
>If we mutines is my lack of yourself?
>
>PERDITA:
>I must greet those than schall death.
>If though first that dearne death.
>Good, ho! mean the did she many like
>That she weak of not a secontent to heard,
>Nor saint the gorand recons thee gace.
>Musicianst they wife this remaining ricused laugh.
>A I doubt alloving to peropace from meet:
>And the dight our fortus Rostain our part not.
>Now may counself dight. Duke of Will, which I say.
>What from Hersend and do confess, for their end.
>In shamelly we from he neive to man,
>To may record: thereful we for dest friend,
>And someting against his prize blow:
>Sumpherdents not his friend; and with ourabes
>That shall shep reved: there thou at best straies
>I we homend to himself, blood have from thee death.
>There then remembred ment shouldst ruch thou death,
>Af such a pleast of the greenst of the acreets,
>Cut the was all, whold for my last plain.
>Juliet: the conse recourted pleasure so him:
>Good the acrouss their strength lature and colar,-upon
>I know not be pruck'd and min--
>Citizens: where's forth will bear not to men
>To wine their recourse, yet parlies,
>Belam his grince his ressorved him.
>
>QUEEN MARGARET:
>Thou cursed we pressalers in the good of long.
