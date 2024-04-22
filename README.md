# From Generalized Go to Lengo
mainly completed by jzb.

This article attempts to describe and generalize the rules of Go using differential geometry. This type of promotion includes:
* Extend the manifold of Go to manifolds other than the Cartesian plane.
* Promote the state space of Go (black, white, empty) from discrete to continuous form.
* Promote the board space of Go from discrete to continuous form.
* Promote the time steps of Go from discrete to continuous form.

Among them, the first type of promotion is easy to complete.
For example, Go defined in a hexagonal or other polygonal paving - which elementary school students can construct, and Go defined in a three-dimensional or high-dimensional lattice.
Furthermore, by slightly modifying the connectivity of the original chessboard (by sticking the edges together), one can obtain a topologically modified chessboard equivalent to torus, Mobius bands, Klein bottle faces, donut faces, and real projective planes, which are widely known (see Wikipedia).
However, there has been no exploration of the continuous promotion in state space, chessboard space, time steps, and other aspects.
Meanwhile, the widespread promotion lacks a unified theoretical framework, which limits the freedom of promotion.
The differential geometry tools used in this article are mainly inspired by research on cellular automata, discrete differential geometry, and manifold learning in machine learning.

The connection between the rules of Go and cellular automata has been noticed for a long time, and some have even extended this connection to develop Go of Life and other games.
For example, the neighborhood (up, down, left, right) used in determining qi in Go is referred to as the "von Neumann neighborhood" in cellular automaton literature.
Like Go, classical cellular automata (such as Conway's Life Game and Wolfram's Rule) have the characteristics of triple discretization in space, state space, and time steps.
Cellular automata, as a numerical simulation tool, have a profound connection with partial differential equations. The choice between discretization and continuity is an important topic in the promotion research of cellular automata.
Once the triple discretization of cellular automata becomes triple continuous, it becomes identical to partial differential equations, which can be referred to as the Lenia project.

Meanwhile, the selection of manifolds is also a major theme in the promotion of cellular automata.
The selection of polygonal paving (even Penrose paving/Voronoi lattice), high-dimensional, and heterotopological surfaces is as ordinary as in Go. (These are common options provided in the CA simulation software Golly).
Google's Mordvintsev et al. combined cellular automata with neural network algorithms to create neuronal cellular automata (NCA) for image processing.
They used gradients and Laplacian operators on the image as input feature functions for the neural network, which also coincides with the partial differential equation interpretation of cellular automata.
The manifold learning theory also tells us that in graph theory, graphs, polygonal lattices, etc., can be regarded as discretization of a certain Riemannian manifold, and the graph Laplace operator of these things has the same properties as the corresponding continuous Laplace operator.

The theory of manifold learning and cellular automata has established the duality between discrete and continuous, and we will use these tools to deal with similar situations in Go.
The basic principle of this continuity is that when using a simple forward integration of the Euler method with a step size of 1 on the system, it can roughly restore the familiar rules of Go.