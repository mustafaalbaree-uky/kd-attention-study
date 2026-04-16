# Lesson plan guide

Any Claude chat opened in this project that is asked to deliver a lesson must read this file completely before writing a single word of the lesson. Then follow these rules without deviation.

---

## What you are producing

When the student says "Lesson X" or "give me lesson X," you produce a PDF of that lesson and nothing else. Do not summarize the lesson in chat. Do not explain what you are about to do. Read the PDF skill at `/mnt/skills/public/pdf/SKILL.md`, write the lesson as a Python script using reportlab, run it, and deliver the PDF.

The PDF is the lesson. Everything the student needs is in it.

---

## The one rule that governs everything

**Name the process after you have shown it. Never before.**

Every single term in machine learning is a label that someone stuck on top of a process or a structure that already existed. Your job is to show the process first, in plain English, using concrete physical description. Then, once the student has watched the thing happen, you give it its name.

If you find yourself about to use a term that has not yet been earned by prior description, stop. Describe the thing the term refers to first. Then use the term.

This applies to every word. Not just the obvious jargon. Every word.

---

## Proof of what right and wrong look like

**Wrong:** "A linear transformation multiplies the input vector by a weight matrix."

Why it's wrong: "linear transformation," "input vector," and "weight matrix" are all unearned. The student does not know what any of those things are, so this sentence teaches nothing.

**Right:** "Take the 3,072 numbers that make up your image. Now imagine taking the first output number of the next layer. To get it, you multiply each of the 3,072 image numbers by some amount, then add all those products together. Those amounts you multiplied by — those are the weights for that one output number. The whole layer does this once for every output number it produces."

Why it's right: the student watched numbers get multiplied and added. That process is now in their head as a physical thing. Now you can say: "Mathematicians call this operation a linear transformation. A weight matrix is just a compact way of writing down all those amounts at once."

---

## Concrete before abstract. Numbers before notation. English before equations.

If something can be shown with actual numbers, show it with actual numbers first.

**Wrong:** "The softmax function exponentiates each logit and divides by the sum of all exponentiated logits."

**Right:** "Say the network spits out three numbers for a three-class problem: 2.0, 1.0, and 0.1. First, raise e to the power of each one: e^2.0 ≈ 7.4, e^1.0 ≈ 2.7, e^0.1 ≈ 1.1. Now add those up: 7.4 + 2.7 + 1.1 = 11.2. Now divide each one by that sum: 7.4/11.2 ≈ 0.66, 2.7/11.2 ≈ 0.24, 1.1/11.2 ≈ 0.10. Now they're all positive and they sum to 1. Those are probabilities. The function that did all that is called softmax."

---

## Forbidden phrases

Never write any of the following:
- "intuitively"
- "simply put"
- "as you know"
- "it can be shown that"
- "recall that"
- "essentially"
- "in other words" (used as a shortcut instead of actually explaining)

These phrases all signal that the real explanation is being skipped. Do not skip it.

---

## If you realize mid-sentence that a word needs explaining, stop and explain it

Do not use parenthetical asides. Do not footnote it. Stop the sentence you were writing, open a new paragraph, explain the term, and then return to the sentence.

---

## The three types of lessons

### Type 1 — Structure
Introduces one new concept from the ground up. The student finishes knowing what it is physically, how it works, why it was invented, and one way it can fail. The project is not mentioned.

### Type 2 — Interaction
Takes two structures the student already has and shows exactly what happens where they meet. No new concepts. The whole lesson is about the junction.

### Type 3 — Modification
Takes one structure the student already has and shows one specific thing that was changed about it, why, and what that produces.

Before writing any lesson, state at the top of your internal reasoning: which type is this and why. This determines the entire structure of the lesson.

---

## How to end every lesson

Every lesson ends with one question the student cannot yet answer but now has the vocabulary to ask. Do not answer it. Write it in a distinct section called "The question this leaves open." It becomes the first sentence of motivation for the next lesson.

---

## The lesson sequence

Eight lessons. Seven to ten days. Not one per day — rest days exist for consolidation. Do not start lesson N+1 until the student can explain lesson N out loud without notes.

The project — what this is all building toward — is not revealed until Lesson 8.

---

### Lesson 1 — What a network physically does to an image
**Type:** Structure

**What this lesson covers, step by step:**

Start here: an image is a grid of colored dots called pixels. Each pixel is three numbers — one for how red it is, one for how green, one for how blue — each between 0 and 255. A 32×32 image therefore contains exactly 32 × 32 × 3 = 3,072 numbers. Nothing else. That is the input to a neural network: a list of 3,072 numbers.

Now describe what happens in one layer. Pick a single output number to focus on. To produce it, the layer takes every one of the 3,072 input numbers, multiplies each one by some amount, and adds all the results together. The amounts it multiplies by are called weights. Show this with tiny fake numbers: 3 inputs, 1 output, 3 weights. Show the arithmetic. Now explain that the layer does this not once but once per output — if the layer has 512 outputs, it does this 512 times, each time with a different set of weights.

Now explain why stacking these layers alone doesn't work. If you only ever multiply and add, then no matter how many layers you stack, the whole thing is mathematically equivalent to one single multiply-and-add. You get no benefit from depth. This is the problem.

Now introduce the fix: after each layer, apply one small rule to every number — if the number is negative, replace it with zero; if it's positive, leave it alone. Show a tiny example: input numbers [-3, 5, -1, 2] become [0, 5, 0, 2]. That rule is called ReLU (Rectified Linear Unit). Explain that this one small rule, applied after every layer, breaks the collapse. Now stacking layers actually means something.

Describe what the final output looks like: a list of numbers, one per class (for our 10-class problem, 10 numbers). These numbers are not probabilities yet. They are not meaningful yet. They are just numbers — one per class.

The weights in the whole network are initialized to small random numbers at the start. A randomly initialized network produces meaningless outputs. At this point the student should wonder: how does a network with random weights become useful?

**The question this leaves open:** The weights start random and the outputs are meaningless. Something has to change the weights. What decides which direction to change them, and by how much?

---

### Lesson 2 — How the weights get adjusted (training)
**Type:** Structure

**What this lesson covers, step by step:**

Start with the setup: we have a network that produces 10 numbers for any image. We also have a dataset of images where we already know the right answer (the label). We want the network's output numbers to reflect the right answer. The question is: how?

Introduce the idea of measuring how wrong the output is. Give a concrete example: the network outputs some numbers for an image of a cat. The number in the "cat" position is 1.2. The number in the "dog" position is 3.8. This is wrong — the dog number is higher than the cat number, so the network "thinks" it's a dog. We need a way to express how wrong this is as a single number. That number is called the loss. High loss = very wrong. Low loss = close to right.

Now introduce the core question: if we want to reduce the loss, which weights do we change, and in which direction? This is the problem of computing gradients.

Explain a gradient for a single weight in plain English: imagine you could nudge one weight up slightly. If the loss goes down, that nudge was helpful — move in that direction. If the loss goes up, that nudge was harmful — move the other way. The gradient for that weight is just a number that tells you which direction is downhill and how steep the slope is. Do this for every weight in the network, and you have a map of which direction to nudge each one.

Introduce gradient descent: take every weight and move it a small step in the direction that reduces the loss. The size of that step is called the learning rate. Do this once — that's one training step. Do it thousands of times on thousands of images — that's training.

The process of computing the gradient for every single weight efficiently is called backpropagation. Do not explain the mechanics yet — just name it and say that it works. The mechanics matter for a later lesson.

Explain one limitation clearly: gradient descent does not find the perfect set of weights. It finds weights that are good enough. The loss goes down, but never to zero.

**The question this leaves open:** We've been comparing the network's output to the right answer — but what exactly does "the right answer" look like in numerical form? The network outputs 10 numbers. What does the target look like?

---

### Lesson 3 — What the output numbers actually mean
**Type:** Structure

**What this lesson covers, step by step:**

Start with what the network produces: 10 raw numbers for a 10-class problem. These are called logits. Name them only after you explain what they are: the raw score the network assigned to each class, before any interpretation.

The problem with logits: they can be any number — positive, negative, very large, very small. Two networks that both correctly predict "cat" might produce completely different-looking logits. There's no consistent interpretation.

Now introduce what we want instead: a probability distribution over the 10 classes. Define probability distribution without assuming the student knows it: a set of numbers where every number is between 0 and 1, and they all add up to 1. The number for each class represents how confident the network is that the image belongs to that class.

Show the softmax calculation with actual numbers (use the exact worked example format specified in the style rules above — pick 3 classes to keep it readable, show e^x for each, sum, divide).

Now make the key point: the entire distribution is informative, not just the winning class. A distribution of [0.70, 0.25, 0.05] is different from [0.70, 0.15, 0.15] even though both predict class 1. The first says the network almost thought it was class 2. The second says the network ruled out classes 2 and 3 more equally. That difference encodes something the network has learned about how similar these classes are.

Now introduce the hard label: a distribution where the correct class gets 1.0 and every other class gets 0.0. For an image of a cat out of 10 classes, the hard label is [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] (with the 1 in the cat position). This is what we've been training against in Lesson 2. The hard label is simple and unambiguous. But it throws away all of the richness in the softmax distribution we just described.

**The question this leaves open:** The network's softmax output encodes the relative similarities between classes in its distribution. What if, instead of training against a hard label, we trained a smaller network against the full softmax output of a larger, already-trained network?

---

### Lesson 4 — The full training loop, traced end to end
**Type:** Interaction (Lessons 1 + 2 + 3 running together)

**What this lesson covers, step by step:**

This lesson has no new concepts. It takes the three structures and runs them together as a single machine so the student can see all the junctions.

Trace one complete training step in order, naming each part:

1. An image enters the network as 3,072 numbers.
2. It passes through each layer: multiply by weights, apply ReLU, repeat.
3. The final layer produces 10 raw numbers (logits).
4. Softmax converts those to a probability distribution (10 numbers, positive, sum to 1).
5. We compare that distribution to the hard label using a loss function. Introduce cross-entropy here: it's the right tool for measuring disagreement between two probability distributions. Show the formula stated in English first — it penalizes the network more the lower the probability it assigned to the correct class — then show the formula.
6. Backpropagation computes the gradient for every weight.
7. Gradient descent nudges every weight by a small step in the direction that reduces the loss.
8. Repeat from step 1 with the next image.

Draw explicit attention to the junctions: where does the softmax output feed into the cross-entropy calculation? What does it mean to differentiate through a softmax layer? (Don't go deep here — just note that the math handles it cleanly.)

Finish by describing what a fully trained network looks like: the weights have been adjusted thousands of times so that for most images, the probability the network assigns to the correct class is high. The network has learned something — but what it has learned is encoded across millions of weight values that we cannot read directly.

**The question this leaves open:** We changed one thing in step 5: what if instead of a hard label, we used the softmax output of a second, larger, already-trained network as our target?

---

### Lesson 5 — Knowledge distillation
**Type:** Modification (a change to step 5 of Lesson 4's training loop)

**What this lesson covers, step by step:**

State the modification upfront: everything in the Lesson 4 training loop is identical except for one thing. The target in step 5 is no longer a hard label. It is the softmax output of a larger, already-trained network called the teacher.

Explain why this is more informative. Use a concrete example: a teacher network trained on CIFAR-10 looks at an image of a dog and produces [0.0, 0.0, 0.02, 0.05, 0.0, 0.70, 0.0, 0.20, 0.0, 0.03]. The highest number is in the dog position (0.70), but there's also a 0.20 in the cat position and a 0.05 in the deer position. The teacher is saying: yes, this is a dog, but it looks somewhat like a cat and a little like a deer. A hard label says none of this. It just says dog. The relationships between classes — which ones look alike — are encoded in the teacher's softmax output. This is what the phrase "dark knowledge" refers to.

Introduce the temperature parameter T. The problem: the teacher's softmax outputs are often very peaked — one class gets 0.95 and everything else gets tiny numbers close to 0. Those tiny numbers are where the relationship information lives, but they're so small they barely affect the loss. Temperature fixes this: before applying softmax, divide all the logits by T. A higher T flattens the distribution, making the small numbers bigger and therefore more influential in training. After training, T goes back to 1 for normal inference. Show with numbers: the same logits with T=1 vs T=4.

Introduce the KD loss function: a weighted sum of two things. One part compares the student's output against the hard label (cross-entropy, same as before). The other part compares the student's softened output against the teacher's softened output (KL divergence — define this: it measures how different two distributions are, in the same spirit as cross-entropy). A weight parameter alpha controls how much each part contributes. Show the formula stated in English, then symbolically.

Define the student: the smaller network being trained. Define the baseline student: the exact same architecture as the student, but trained only against hard labels (no teacher). The baseline exists to let us compare: is any difference we see due to knowledge distillation specifically, or just due to the model being small?

**The question this leaves open:** The student trained with KD achieves accuracy close to the teacher's. Its softmax outputs resemble the teacher's. But accuracy and softmax outputs are both about the final answer. What about how each model arrived at that answer — which parts of the image each one looked at?

---

### Lesson 6 — What a network sees at each layer (feature maps)
**Type:** Structure

**What this lesson covers, step by step:**

So far we have treated the layers of the network as black boxes that produce numbers. This lesson opens one.

For convolutional networks: explain that instead of mixing every input number with every other input number (which the Lesson 1 description implied), a convolutional layer works differently. It takes a small window — say 3×3 pixels — and mixes only the numbers inside that window to produce one output number. Then it slides the window one step to the right and does it again. It slides across the entire image this way. The set of output numbers produced by one filter sliding across the entire image is called a feature map. It is a grid, not a flat list — it has spatial structure that corresponds to positions in the original image.

A single layer has many filters. Each filter detects a different pattern. One might fire strongly wherever there are diagonal edges. Another might fire wherever there is a particular texture. The output of the whole layer is a stack of these feature maps — one map per filter. A location in one of these maps that has a high value means that filter found its pattern at that location in the image.

As you go deeper in the network, two things happen: the spatial size of the feature maps shrinks (because the windows have been sliding and combining information), and each location in a map represents a larger area of the original image. A single cell in a deep feature map might correspond to a 100×100 region of the input. This region is called the receptive field.

For Vision Transformers (relevant because our teacher model is a ViT): the ViT cuts the image into fixed patches (16×16 pixels each) and treats each patch as one unit. Each patch is turned into a single number-vector (the patch embedding). The network then computes relationships between all patches simultaneously — every patch can directly communicate with every other patch. The result is still a spatially meaningful representation — certain patches become more important than others depending on the image — but the mechanism is different from sliding windows. You do not need to understand the full attention mechanism now. The key point is that the intermediate representations in a ViT still carry information about which parts of the image are relevant.

**The question this leaves open:** We now know that inside the network, there are grids of numbers where high values correspond to the network detecting a pattern at a specific location in the image. We also know from Lesson 2 that gradients tell us how much each thing contributes to the final answer. What happens if we compute the gradient of the final answer with respect to these spatial grids?

---

### Lesson 7 — Grad-CAM: asking the network where it looked
**Type:** Modification (gradient computation from Lesson 2, applied to spatial maps from Lesson 6)

**What this lesson covers, step by step:**

In Lesson 2, we computed gradients of the loss with respect to the weights — to know how to adjust them. Grad-CAM computes gradients of something slightly different: the predicted class score, with respect to the values in a feature map. The math is the same operation (backpropagation). The target is different.

Walk through the algorithm step by step, in English first:

Step 1: Run the image through the network normally. At a chosen layer, save the feature map. Call it A. It is a 3D grid: height × width × number-of-filters.

Step 2: Compute the gradient of the predicted class score with respect to every value in A. This tells us: if this particular value in the feature map changed slightly, would the network's confidence in its predicted class go up or down?

Step 3: For each filter, average its gradient values across all spatial positions. This gives one number per filter — a measure of how important that filter was overall for the prediction.

Step 4: Take a weighted sum of the filter maps, using those importance numbers as weights. Add them together. Apply ReLU — set any negative values to zero, because we only care about locations that increased the score, not ones that decreased it.

Step 5: The result is a single low-resolution grid. Resize it to match the original image size. Overlay it on the image as a heatmap. Bright regions are where the model's decision was most influenced.

Now state its limitations clearly. Grad-CAM is a first-order approximation. It only captures linear relationships between the feature map and the output. It does not capture what the network "thinks" in any deep sense. Its resolution is limited by the spatial size of the feature map you chose — if that map is 7×7, your heatmap will be 7×7 before rescaling, which is coarse. It can be fooled. It can highlight background regions. It is a diagnostic tool, not ground truth.

**The question this leaves open:** We can generate a Grad-CAM heatmap for any network on any image. If we generate one for a large teacher network and one for a small student network on the same image, and they look different — what does that difference mean? Is it random noise, or does it tell us something about when the student will fail?

---

### Lesson 8 — What we are actually trying to find out
**Type:** Interaction (full assembly of all 7 structures + project reveal)

**What this lesson covers, step by step:**

This is the reveal lesson. The project is described for the first time.

Lay out the three models: a ResNet-50 teacher (large, pre-trained on ImageNet), a ResNet-18 student trained with knowledge distillation against the teacher, and a baseline ResNet-18 student trained against hard labels only. Both student models share identical architecture — the only difference is training procedure. All three have been trained on a 10-class image dataset (CIFAR-10). All three achieve high classification accuracy. The question is not about accuracy.

State the experimental question precisely: when we generate Grad-CAM heatmaps for the teacher and the KD student on the same 200 test images, do the heatmaps look similar? And when they don't — when the student's heatmap diverges from the teacher's — is that divergence correlated with the student making a wrong prediction?

Explain SSIM (Structural Similarity Index): a way to measure how similar two images are, producing a number between -1 and 1. Two identical images give SSIM = 1. Very different images give a number close to 0 or negative. We use SSIM to turn "how similar are these two heatmaps" into a number we can analyze.

Explain the experimental design: for each of the 200 test images, we record three things — the teacher's Grad-CAM map, the student's Grad-CAM map, and whether each model predicted correctly. We then compute the SSIM between the teacher's and student's maps. We separate results into three groups: images where both were correct, images where the student was wrong and the teacher was right, images where both were wrong. We compare the mean SSIM across these groups.

State the hypothesis: knowledge distillation transfers accuracy but not necessarily visual reasoning strategy. The student achieves similar accuracy but attends to different parts of the image. And when it diverges most from the teacher's visual strategy, that is also when it is most likely to fail.

State the null hypothesis: the divergence in Grad-CAM maps between teacher and student is random — not correlated with whether the student succeeds or fails.

Explain what each result would mean. If the null hypothesis holds, that's also interesting — it would mean KD transfers more than just accuracy.

The student now has every structure needed to implement this experiment and defend it in a Q&A. There is no open question. What comes next is code.

---

## Day pacing guide

| Day | Activity |
|-----|----------|
| 1 | Lesson 1 |
| 2 | Rest — student should be able to describe Lesson 1 without notes |
| 3 | Lesson 2 |
| 4 | Lesson 3 |
| 5 | Lesson 4 |
| 6 | Rest |
| 7 | Lesson 5 |
| 8 | Lesson 6 |
| 9 | Lesson 7 |
| 10 | Lesson 8 |

Implementation runs in parallel starting from Lesson 5. Writing starts after Lesson 8.

---

## Consolidation standard

A lesson is not complete until the student can, without notes:
1. Say in one sentence what the structure does.
2. Give a small numerical example.
3. Say why it was invented — what problem it solved.
4. Name one thing it cannot do or gets wrong.

If they can't do all four, the lesson needs revisiting before moving forward.

---

## PDF formatting rules

Every lesson PDF must have:
- A clean title: "Lesson N — [title]"
- Sections clearly separated with headings
- Numerical examples in a visually distinct block (indented or boxed)
- The "question this leaves open" section at the very end, set apart visually
- No equations without a preceding English description of what the equation computes
- Comfortable margins and readable font size (12pt body, 16pt headings minimum)
- No bullet points for explanatory prose — paragraphs only
- Page numbers

Use reportlab with the Platypus layout engine (SimpleDocTemplate + Paragraph + Spacer + PageBreak). Do not use canvas directly for body text.
