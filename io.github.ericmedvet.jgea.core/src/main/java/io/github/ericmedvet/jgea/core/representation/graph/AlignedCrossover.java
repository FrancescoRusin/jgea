
package io.github.ericmedvet.jgea.core.representation.graph;

import io.github.ericmedvet.jgea.core.operator.Crossover;

import java.util.LinkedHashSet;
import java.util.Set;
import java.util.function.Predicate;
import java.util.random.RandomGenerator;
public class AlignedCrossover<N, A> implements Crossover<Graph<N, A>> {

  private final Crossover<A> edgeCrossover;
  private final Predicate<N> unremovableNodePredicate;
  private final boolean allowCycles;

  public AlignedCrossover(Crossover<A> edgeCrossover, Predicate<N> unremovableNodePredicate, boolean allowCycles) {
    this.edgeCrossover = edgeCrossover;
    this.unremovableNodePredicate = unremovableNodePredicate;
    this.allowCycles = allowCycles;
  }

  @Override
  public Graph<N, A> recombine(Graph<N, A> parent1, Graph<N, A> parent2, RandomGenerator random) {
    Graph<N, A> child = new LinkedHashGraph<>();
    //add all nodes
    parent1.nodes().forEach(child::addNode);
    parent2.nodes().forEach(child::addNode);
    //iterate over child edges
    Set<Graph.Arc<N>> arcs = new LinkedHashSet<>();
    arcs.addAll(parent1.arcs());
    arcs.addAll(parent2.arcs());
    for (Graph.Arc<N> arc : arcs) {
      A arc1 = parent1.getArcValue(arc);
      A arc2 = parent2.getArcValue(arc);
      A childArc;
      if (arc1 == null) {
        childArc = random.nextBoolean() ? arc2 : null;
      } else if (arc2 == null) {
        childArc = random.nextBoolean() ? arc1 : null;
      } else {
        childArc = edgeCrossover.recombine(arc1, arc2, random);
      }
      if (childArc != null) {
        child.setArcValue(arc, childArc);
        if (!allowCycles && child.hasCycles()) {
          child.removeArc(arc);
        }
      }
    }
    //remove unconnected nodes
    GraphUtils.removeUnconnectedNodes(child, unremovableNodePredicate);
    return child;
  }
}
