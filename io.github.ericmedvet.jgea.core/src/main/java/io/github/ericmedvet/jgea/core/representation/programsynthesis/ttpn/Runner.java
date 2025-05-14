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
package io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn;

import io.github.ericmedvet.jgea.core.representation.programsynthesis.InstrumentedProgram;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.ProgramExecutionException;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.RunProfile;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.Type;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Runner {

  private final int maxNOfSteps;
  private final int maxNOfTokens;
  private final int maxTokensSize;
  private final int maxSingleTokenSize;
  private final boolean skipExecutionOfBlockedNetworks;

  public interface TTPNInstrumentedOutcome extends InstrumentedProgram.InstrumentedOutcome {
    record Key(Wire wire, int step) {}

    Network network();

    Map<Key, List<Object>> wireContents();

    static TTPNInstrumentedOutcome from(
        List<Object> outputs,
        ProgramExecutionException exception,
        Network network,
        Map<Key, List<Object>> wireContents
    ) {
      int steps = wireContents.keySet().stream().mapToInt(Key::step).max().orElse(0);
      List<RunProfile.State> states = IntStream.range(0, steps + 1)
          .mapToObj(
              k -> wireContents.entrySet()
                  .stream()
                  .filter(e -> e.getKey().step == k)
                  .collect(Collectors.groupingBy(e -> network.concreteOutputType(e.getKey().wire().src())))
          )
          .map(
              m -> new RunProfile.State(
                  m.entrySet()
                      .stream()
                      .collect(
                          Collectors.toMap(
                              Map.Entry::getKey,
                              oe -> oe.getValue().stream().mapToInt(e -> e.getValue().size()).sum()
                          )
                      ),
                  m.entrySet()
                      .stream()
                      .collect(
                          Collectors.toMap(
                              Map.Entry::getKey,
                              oe -> oe.getValue()
                                  .stream()
                                  .mapToInt(e -> e.getValue().stream().mapToInt(o -> oe.getKey().sizeOf(o)).sum())
                                  .sum()
                          )
                      )
              )
          )
          .toList();
      record HardTTPNInstrumentedOutcome(
          List<Object> outputs, ProgramExecutionException exception, RunProfile profile, Network network,
          Map<Key, List<Object>> wireContents
      ) implements TTPNInstrumentedOutcome {}
      return new HardTTPNInstrumentedOutcome(
          outputs,
          exception,
          new RunProfile(states),
          network,
          wireContents
      );
    }

    static TTPNInstrumentedOutcome from(List<Object> outputs, Network network, Map<Key, List<Object>> wireContents) {
      return from(outputs, null, network, wireContents);
    }

    static TTPNInstrumentedOutcome from(
        ProgramExecutionException exception,
        Network network,
        Map<Key, List<Object>> wireContents
    ) {
      return from(null, exception, network, wireContents);
    }

    static TTPNInstrumentedOutcome from(ProgramExecutionException exception, Network network) {
      return from(null, exception, network, Map.of());
    }
  }

  public Runner(
      int maxNOfSteps,
      int maxNOfTokens,
      int maxTokensSize,
      int maxSingleTokenSize,
      boolean skipExecutionOfBlockedNetworks
  ) {
    this.maxNOfSteps = maxNOfSteps;
    this.maxNOfTokens = maxNOfTokens;
    this.maxTokensSize = maxTokensSize;
    this.maxSingleTokenSize = maxSingleTokenSize;
    this.skipExecutionOfBlockedNetworks = skipExecutionOfBlockedNetworks;
  }

  private static <T> Queue<T> emptyQueue() {
    return new ArrayDeque<>();
  }

  private static <T> List<T> takeAll(Queue<T> queue) {
    List<T> list = new ArrayList<>(queue);
    queue.clear();
    return list;
  }

  private static <T> List<T> takeExactly(Queue<T> queue, int n) {
    return IntStream.range(0, n).mapToObj(i -> queue.remove()).toList();
  }

  private static <T> Optional<T> takeOne(Queue<T> queue) {
    if (queue.isEmpty()) {
      return Optional.empty();
    }
    return Optional.of(queue.remove());
  }

  public InstrumentedProgram asInstrumentedProgram(Network network) {
    Runner thisRunner = this;
    return new InstrumentedProgram() {
      @Override
      public InstrumentedOutcome runInstrumented(List<Object> inputs) {
        return thisRunner.run(network, inputs);
      }

      @Override
      public List<Type> inputTypes() {
        return network.inputGates().values().stream().toList();
      }

      @Override
      public List<Type> outputTypes() {
        return network.outputGates().values().stream().toList();
      }

      @Override
      public String toString() {
        return "ttpn[g=%d,w=%d]".formatted(network.gates().size(), network.wires().size());
      }
    };
  }

  public TTPNInstrumentedOutcome run(Network network, List<Object> inputs) {
    // check validity
    List<Type> inputTypes = network.inputGates().values().stream().toList();
    SortedMap<Integer, Type> outputTypes = network.outputGates();
    if (inputs.size() != inputTypes.size()) {
      return TTPNInstrumentedOutcome.from(
          new ProgramExecutionException(
              "Wrong number of inputs: %d expected, %d found".formatted(
                  inputTypes.size(),
                  inputs.size()
              )
          ),
          network
      );
    }
    for (int i = 0; i < inputTypes.size(); i++) {
      if (!inputTypes.get(i).matches(inputs.get(i))) {
        return TTPNInstrumentedOutcome.from(
            new ProgramExecutionException(
                "Invalid input type for input %d of type %s: %s".formatted(
                    i,
                    inputTypes.get(i),
                    inputs.get(i).getClass()
                )
            ),
            network
        );
      }
    }
    // check if blocked
    if (skipExecutionOfBlockedNetworks) {
      long nOfBlockedOutputs = network.outputGates()
          .keySet()
          .stream()
          .filter(gi -> network.isDeadGate(gi) || !network.isWiredToInput(gi))
          .count();
      if (nOfBlockedOutputs > 0) {
        return TTPNInstrumentedOutcome.from(
            new ProgramExecutionException(
                "Blocked/unwired output gates: %d on %d".formatted(nOfBlockedOutputs, network.outputGates().size())
            ),
            network
        );
      }
    }
    // prepare memory
    Map<Wire, Queue<Object>> current = new HashMap<>();
    network.wires().forEach(w -> current.put(w, new ArrayDeque<>()));
    Map<Wire, List<Object>> next = new HashMap<>();
    Map<Wire, Type> actualTypes = new HashMap<>();
    Map<Integer, Object> inputsMap = new HashMap<>();
    int ii = 0;
    for (Integer inputGI : network.inputGates().keySet()) {
      inputsMap.put(inputGI, inputs.get(ii));
      ii = ii + 1;
    }
    for (Wire w : network.wires()) {
      Type type = network.concreteOutputType(w.src());
      if (type == null) {
        return TTPNInstrumentedOutcome.from(
            new ProgramExecutionException("No concrete type at output port %s".formatted(w.src())),
            network
        );
      }
      actualTypes.put(w, type);
    }
    // prepare state, counter, and output map
    int k = 0;
    SortedMap<Integer, Object> outputs = new TreeMap<>();
    Map<TTPNInstrumentedOutcome.Key, List<Object>> wireContents = new HashMap<>();
    // iterate on time steps
    while (k < maxNOfSteps) {
      // keep track of wire contents
      int finalK = k;
      current.forEach(
          (w, tokens) -> wireContents.put(
              new TTPNInstrumentedOutcome.Key(w, finalK),
              new ArrayList<>(tokens)
          )
      );
      // iterate on gates
      for (Map.Entry<Integer, Gate> gateEntry : network.gates().entrySet()) {
        int gi = gateEntry.getKey();
        Gate g = gateEntry.getValue();
        if (g instanceof Gate.InputGate) {
          if (k == 0) {
            network.wiresFrom(gi, 0).forEach(w -> next.put(w, List.of(inputsMap.get(gi))));
          }
        } else if (g instanceof Gate.OutputGate) {
          network.wireTo(gi, 0).flatMap(w -> takeOne(current.get(w))).ifPresent(token -> outputs.put(gi, token));
        } else {
          List<Queue<Object>> inputQueues = IntStream.range(0, g.inputPorts().size())
              .mapToObj(
                  pi -> network.wireTo(
                      gi,
                      pi
                  ).map(current::get).orElse(emptyQueue())
              )
              .toList();
          // check conditions and possibly apply function
          if (IntStream.range(0, g.inputPorts().size())
              .allMatch(
                  pi -> inputQueues.get(pi).size() >= g.inputPorts()
                      .get(
                          pi
                      )
                      .n()
              )) {
            Gate.Data localIn = Gate.Data.of(
                IntStream.range(0, g.inputPorts().size())
                    .mapToObj(pi -> switch (g.inputPorts().get(pi).condition()) {
                      case EXACTLY -> takeExactly(inputQueues.get(pi), g.inputPorts().get(pi).n());
                      case AT_LEAST -> takeAll(inputQueues.get(pi));
                    })
                    .toList()
            );
            try {
              Gate.Data localOut = g.operator().apply(localIn);
              // check number of outputs
              if (localOut.lines().size() != g.outputTypes().size()) {
                return TTPNInstrumentedOutcome.from(
                    new ProgramExecutionException(
                        "Unexpected wrong number of outputs: %d expected, %d found".formatted(
                            g.outputTypes()
                                .size(),
                            localOut.lines().size()
                        )
                    ),
                    network,
                    wireContents
                );
              }
              // check output size
              for (int pi = 0; pi < g.outputTypes().size(); pi++) {
                Type type = network.concreteOutputType(new Wire.EndPoint(gi, pi));
                if (type != null && !type.isGeneric()) {
                  for (Object token : localOut.all(pi)) {
                    if (type.sizeOf(token) > maxSingleTokenSize) {
                      return TTPNInstrumentedOutcome.from(
                          new ProgramExecutionException(
                              "Exceeded size of token of type %s on gate %s: %d > %d".formatted(
                                  type,
                                  g,
                                  type.sizeOf(token),
                                  maxSingleTokenSize
                              )
                          ),
                          network,
                          wireContents
                      );
                    }
                  }
                }
              }
              // put outputs
              IntStream.range(0, localOut.lines().size())
                  .forEach(
                      pi -> network.wiresFrom(gi, pi)
                          .forEach(w -> next.put(w, localOut.lines().get(pi)))
                  );
            } catch (RuntimeException e) {
              return TTPNInstrumentedOutcome.from(
                  new ProgramExecutionException("Cannot run %s on %s due to %s".formatted(g, localIn, e.toString()), e),
                  network,
                  wireContents
              );
            }
          }
        }
      }
      // check no new tokens
      if (next.values().stream().mapToInt(List::size).sum() == 0) {
        break;
      }
      // add tokens to current
      next.forEach((w, ts) -> current.get(w).addAll(ts));
      next.clear();
      // check conditions
      int nOfTokens = current.values().stream().mapToInt(Queue::size).sum();
      if (nOfTokens > maxNOfTokens) {
        return TTPNInstrumentedOutcome.from(
            new ProgramExecutionException(
                "Exceeded number of tokens: %d > %d".formatted(nOfTokens, maxNOfTokens)
            ),
            network,
            wireContents
        );
      }
      int tokensSize = current.entrySet()
          .stream()
          .mapToInt(e -> {
            Type type = actualTypes.get(e.getKey());
            return e.getValue().stream().mapToInt(type::sizeOf).sum();
          })
          .sum();
      if (tokensSize > maxTokensSize) {
        return TTPNInstrumentedOutcome.from(
            new ProgramExecutionException(
                "Exceeded size of tokens: %d > %d".formatted(tokensSize, maxTokensSize)
            ),
            network,
            wireContents
        );
      }
      // increment k
      k = k + 1;
    }
    // check output
    if (!outputTypes.keySet().equals(outputs.keySet())) {
      return TTPNInstrumentedOutcome.from(
          new ProgramExecutionException(
              "Missing outputs on gates: %s".formatted(
                  outputTypes.keySet()
                      .stream()
                      .filter(gi -> !outputs.containsKey(gi))
                      .map(gi -> Integer.toString(gi))
                      .collect(
                          Collectors.joining(",")
                      )
              )
          ),
          network,
          wireContents
      );
    }
    for (int gi : outputTypes.keySet()) {
      if (!outputTypes.get(gi).matches(outputs.get(gi))) {
        return TTPNInstrumentedOutcome.from(
            new ProgramExecutionException(
                "Invalid output type for input %d of type %s: %s".formatted(
                    gi,
                    outputTypes.get(gi),
                    outputs.get(gi).getClass()
                )
            ),
            network,
            wireContents
        );
      }
    }
    return TTPNInstrumentedOutcome.from(
        outputs.values().stream().toList(),
        network,
        wireContents
    );
  }
}
