/*-
 * ========================LICENSE_START=================================
 * jgea-core
 * %%
 * Copyright (C) 2018 - 2024 Eric Medvet
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =========================LICENSE_END==================================
 */

package io.github.ericmedvet.jgea.core.order;

import io.github.ericmedvet.jgea.core.util.Misc;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class DAGPartiallyOrderedCollection<T> extends AbstractPartiallyOrderedCollection<T> {

  private final Map<Integer, Node<T>> nodes;
  private final Map<T, Integer> contentMap;
  private List<Collection<T>> fronts;

  public DAGPartiallyOrderedCollection(PartialComparator<? super T> partialComparator) {
    super(partialComparator);
    nodes = new LinkedHashMap<>();
    contentMap = new HashMap<>();
    fronts = null;
  }

  public DAGPartiallyOrderedCollection(Collection<? extends T> ts, PartialComparator<? super T> partialComparator) {
    this(partialComparator);
    ts.forEach(this::add);
  }

  private record Node<T>(
      List<T> contents, SequencedSet<Integer> beforeNodes, SequencedSet<Integer> afterNodes
  ) {
  }

  @Override
  public void add(T t) {
    fronts = null;
    for (Map.Entry<Integer, Node<T>> nodeEntry : nodes.entrySet()) {
      PartialComparator.PartialComparatorOutcome outcome = comparator().compare(
          t,
          nodeEntry.getValue().contents().getFirst()
      );
      if (outcome.equals(PartialComparator.PartialComparatorOutcome.SAME)) {
        nodeEntry.getValue().contents().add(t);
        contentMap.put(t, nodeEntry.getKey());
        return;
      }
    }
    Node<T> newNode = new Node<>(new ArrayList<>(List.of(t)), new LinkedHashSet<>(), new LinkedHashSet<>());
    int newNodeIndex = nodes.keySet().stream().max(Integer::compareTo).orElse(0) + 1;
    nodes.forEach((nodeIndex, node) -> {
      PartialComparator.PartialComparatorOutcome outcome = comparator().compare(
          t,
          node.contents().getFirst()
      );
      if (outcome.equals(PartialComparator.PartialComparatorOutcome.BEFORE)) {
        node.beforeNodes().add(newNodeIndex);
        newNode.afterNodes().add(nodeIndex);
      } else if (outcome.equals(PartialComparator.PartialComparatorOutcome.AFTER)) {
        node.afterNodes().add(newNodeIndex);
        newNode.beforeNodes().add(nodeIndex);
      }
    });
    nodes.put(newNodeIndex, newNode);
    contentMap.put(t, newNodeIndex);
  }

  @Override
  public List<Collection<T>> fronts() {
    if (fronts == null) {
      List<Set<Integer>> indexFronts = new ArrayList<>();
      Set<Integer> remaining = new HashSet<>(nodes.keySet());
      while (!remaining.isEmpty()) {
        indexFronts.add(
            remaining.stream()
                .filter(index -> Misc.intersection(nodes.get(index).beforeNodes, remaining).isEmpty())
                .collect(Collectors.toSet())
        );
        remaining.removeAll(indexFronts.getLast());
      }
      fronts = indexFronts.stream()
          .map(
              indexes -> (Collection<T>) indexes.stream()
                  .map(i -> nodes.get(i).contents())
                  .flatMap(Collection::stream)
                  .collect(Collectors.toList())
          )
          .toList();
    }
    return fronts;
  }

  @Override
  public boolean remove(T t) {
    boolean removed = false;
    Integer nodeIndex = contentMap.get(t);
    if (nodeIndex != null) {
      contentMap.remove(t);
      Node<T> node = nodes.get(nodeIndex);
      if (node.contents.contains(t)) {
        fronts = null;
        removed = true;
        node.contents.remove(t);
        if (node.contents.isEmpty()) {
          nodes.remove(nodeIndex);
          node.beforeNodes.forEach(ni -> nodes.get(ni).afterNodes.remove(nodeIndex));
          node.afterNodes.forEach(ni -> nodes.get(ni).beforeNodes.remove(nodeIndex));
        }
      }
    }
    return removed;
  }

  @Override
  public String toString() {
    List<Collection<T>> fronts = fronts();
    return IntStream.range(0, fronts.size())
        .mapToObj(
            i -> "F%d:[%s]".formatted(
                i,
                fronts.get(i).stream().map(Object::toString).collect(Collectors.joining(";"))
            )
        )
        .collect(Collectors.joining(";"));
  }
}
