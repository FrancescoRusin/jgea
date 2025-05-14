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
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.Type;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.TypeException;
import java.util.*;
import java.util.random.RandomGenerator;

public class WireSwapperMutation implements Mutation<Network> {

  private final int maxNOfAttempts;
  private final boolean avoidDeadGates;

  public WireSwapperMutation(int maxNOfAttempts, boolean avoidDeadGates) {
    this.maxNOfAttempts = maxNOfAttempts;
    this.avoidDeadGates = avoidDeadGates;
  }

  @Override
  public Network mutate(Network network, RandomGenerator random) {
    Map<Type, List<Wire>> typedWires = new LinkedHashMap<>();
    network.wires().forEach(w -> {
      Type type = network.concreteInputType(w.dst());
      typedWires.computeIfAbsent(type, t -> new ArrayList<>()).add(w);
    });
    int nOfAttempts = 0;
    List<List<Wire>> lists = typedWires.values().stream().filter(ws -> ws.size() > 1).toList();
    for (List<Wire> wires : lists) {
      for (Wire w1 : wires) {
        for (Wire w2 : wires) {
          if (!w1.dst().equals(w2.dst())) {
            Wire newW1 = new Wire(w1.src(), w2.dst());
            Wire newW2 = new Wire(w2.src(), w1.dst());
            Set<Wire> newWires = new HashSet<>(network.wires());
            newWires.remove(w1);
            newWires.remove(w2);
            newWires.add(newW1);
            newWires.add(newW2);
            try {
              Network newNetwork = new Network(network.gates(), newWires);
              if (!avoidDeadGates || NetworkUtils.deadComparator().compare(newNetwork, network) <= 0) {
                return newNetwork;
              }
            } catch (NetworkStructureException | TypeException e) {
              throw new RuntimeException(e);
            }
            nOfAttempts = nOfAttempts + 1;
            if (nOfAttempts > maxNOfAttempts) {
              return network;
            }
          }
        }
      }
    }
    return network;
  }
}
