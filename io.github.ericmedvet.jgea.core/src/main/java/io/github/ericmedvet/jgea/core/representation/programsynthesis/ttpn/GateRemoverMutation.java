/*-
 * ========================LICENSE_START=================================
 * jgea-core
 * %%
 * Copyright (C) 2018 - 2025 Eric Medvet
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
package io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn;

import io.github.ericmedvet.jgea.core.operator.Mutation;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.TypeException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.random.RandomGenerator;
import java.util.stream.Collectors;

public class GateRemoverMutation implements Mutation<Network> {

  private final int maxNOfAttempts;
  private final boolean avoidDeadGates;

  public GateRemoverMutation(int maxNOfAttempts, boolean avoidDeadGates) {
    this.maxNOfAttempts = maxNOfAttempts;
    this.avoidDeadGates = avoidDeadGates;
  }

  @Override
  public Network mutate(Network network, RandomGenerator random) {
    List<Integer> removableGateIndexes = new ArrayList<>(
        network.gates()
            .entrySet()
            .stream()
            .filter(e -> !(e.getValue() instanceof Gate.InputGate) && !(e.getValue() instanceof Gate.OutputGate))
            .map(Map.Entry::getKey)
            .toList()
    );
    if (removableGateIndexes.isEmpty()) {
      return network;
    }
    int nOfAttempts = 0;
    Collections.shuffle(removableGateIndexes, random);
    for (int toRemoveGi : removableGateIndexes) {
      if (nOfAttempts >= maxNOfAttempts) {
        return network;
      }
      nOfAttempts = nOfAttempts + 1;
      // remove gate
      Network.Deletion deletion = new Network.Deletion(
          List.of(toRemoveGi),
          network.wires()
              .stream()
              .filter(w -> w.src().gateIndex() == toRemoveGi || w.dst().gateIndex() == toRemoveGi)
              .collect(Collectors.toSet())
      );
      try {
        Network newN = deletion.applyTo(network);
        // wire everything
        while (true) {
          Network.Addition addition = NetworkUtils.wire(newN, false, random);
          if (addition.isEmpty()) {
            break;
          }
          newN = addition.applyTo(newN);
        }
        if (!avoidDeadGates || NetworkUtils.deadComparator().compare(newN, network) <= 0) {
          return newN;
        }
      } catch (NetworkStructureException | TypeException e) {
        // ignore
      }
    }
    return network;
  }
}
