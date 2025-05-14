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

import io.github.ericmedvet.jgea.core.representation.programsynthesis.ProgramExecutionException;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.Base;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.Composed;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.Type;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.TypeException;
import io.github.ericmedvet.jgea.core.representation.tree.numeric.Element;
import io.github.ericmedvet.jgea.core.util.IntRange;
import io.github.ericmedvet.jgea.core.util.Misc;
import io.github.ericmedvet.jnb.datastructure.DoubleRange;
import io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction;
import java.util.*;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.random.RandomGenerator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class StatsMain {

  public static final int MAX_N_OF_STEPS = 100;
  public static final int MAX_N_OF_TOKENS = 1000;
  public static final int MAX_TOKENS_SIZE = 2000;
  public static final int MAX_SINGLE_TOKEN_SIZE = 100;

  public static final double XOVER_SUBNET_SIZE_RATE = 0.5;
  public static final List<Gate> ALL_GATES = List.of(
      Gates.rPMathOperator(Element.Operator.MULTIPLICATION),
      Gates.rPMathOperator(Element.Operator.ADDITION),
      Gates.rPMathOperator(Element.Operator.SUBTRACTION),
      Gates.rPMathOperator(Element.Operator.DIVISION),
      Gates.iPMathOperator(Element.Operator.MULTIPLICATION),
      Gates.iPMathOperator(Element.Operator.ADDITION),
      Gates.iPMathOperator(Element.Operator.SUBTRACTION),
      Gates.iPMathOperator(Element.Operator.DIVISION),
      Gates.rSPSum(),
      Gates.rSPMult(),
      Gates.rSSum(),
      Gates.rSMult(),
      Gates.iSPSum(),
      Gates.iSPMult(),
      Gates.iSSum(),
      Gates.iSMult(),
      Gates.splitter(),
      Gates.sSplitter(),
      Gates.pairer(),
      Gates.unpairer(),
      Gates.noop(),
      Gates.length(),
      Gates.iTh(),
      Gates.sequencer(),
      Gates.rToI(),
      Gates.iToR(),
      Gates.sink(),
      Gates.queuer(),
      Gates.equal(),
      Gates.iBefore(),
      Gates.rBefore(),
      Gates.sBefore(),
      Gates.select(),
      Gates.bOr(),
      Gates.bAnd(),
      Gates.bXor(),
      Gates.bNot(),
      Gates.repeater(),
      Gates.iRange(),
      Gates.iConst(0),
      Gates.iConst(1),
      Gates.iConst(5),
      Gates.rConst(0),
      Gates.rConst(0.1),
      Gates.rConst(XOVER_SUBNET_SIZE_RATE),
      Gates.bConst(true)
  );

  public static final DataFactory DATA_FACTORY = new DataFactory(
      List.of(1, 2, 7, 11),
      List.of(1d, 2d, 3d, 1.5, 2.5, 3.14),
      List.of("cat", "dog", "Hello World!", "mummy"),
      new IntRange(-10, 100),
      new DoubleRange(-10, 10),
      new IntRange(2, 20),
      new IntRange(3, 8)
  );

  private static void factoryStats(
      List<Type> inputTypes,
      List<Type> outputTypes,
      int nOfNetworks,
      int maxNumberOfGates,
      int nOfCases,
      RandomGenerator rnd
  ) throws ProgramExecutionException {
    BackTracingNetworkFactory factory = new BackTracingNetworkFactory(
        inputTypes,
        outputTypes,
        new LinkedHashSet<>(ALL_GATES),
        maxNumberOfGates,
        1000
    );
    Runner runner = new Runner(MAX_N_OF_STEPS, MAX_N_OF_TOKENS, MAX_TOKENS_SIZE, MAX_SINGLE_TOKEN_SIZE, false);
    List<List<Object>> cases = IntStream.range(0, nOfCases)
        .mapToObj(
            i -> inputTypes.stream()
                .map(t -> DATA_FACTORY.apply(t, rnd))
                .toList()
        )
        .toList();
    Predicate<Network> goodGatesPredicate = network -> network.gates()
        .keySet()
        .stream()
        .noneMatch(network::isDeadGate);
    List<FormattedNamedFunction<Network, Double>> fs = List.of(
        FormattedNamedFunction.from(n -> (double) n.size(), "%5.1f", "size"),
        FormattedNamedFunction.from(n -> (double) n.gates().size(), "%4.1f", "n.gates"),
        FormattedNamedFunction.from(
            n -> {
              try {
                return (double) (n.disjointSubnetworks().size());
              } catch (NetworkStructureException | TypeException e) {
                return Double.NaN;
              }
            },
            "%4.1f",
            "n.subnetworks"
        ),
        FormattedNamedFunction.from(
            n -> (double) n.inputGates().keySet().stream().filter(n::isWiredToOutput).count(),
            "%3.1f",
            "n.outputWiredInputs"
        ),
        FormattedNamedFunction.from(
            n -> (double) n.outputGates().keySet().stream().filter(n::isWiredToInput).count(),
            "%3.1f",
            "n.inputWiredOutputs"
        ),
        FormattedNamedFunction.from(
            n -> (double) n.deadGates().size(),
            "%4.1f",
            "n.deadGates"
        ),
        FormattedNamedFunction.from(
            n -> (double) n.deadOrIUnwiredOutputGates().size(),
            "%4.1f",
            "n.deadOrIUnwiredOutputGates"
        ),
        FormattedNamedFunction.from(
            n -> goodGatesPredicate.test(n) ? 1d : 0d,
            "*%6.4f",
            "isAllGood"
        ),
        FormattedNamedFunction.from(
            n -> cases.stream()
                .mapToDouble(c -> runner.asInstrumentedProgram(n).safelyRun(c) == null ? 1d : 0d)
                .average()
                .orElseThrow(),
            "*%6.4f",
            "rate.cases.null"
        )
    );
    System.out.println(fs.stream().map(f -> f.name()).collect(Collectors.joining("\t")));
    Function<Collection<Network>, Map<String, Double>> statter = networks -> fs.stream()
        .collect(
            Collectors.toMap(
                f -> f.name(),
                f -> networks.stream().mapToDouble(f::apply).average().orElse(Double.NaN)
            )
        );
    Function<Collection<Network>, String> sStatter = statter.andThen(
        map -> fs.stream()
            .map(f -> f.format().formatted(map.get(f.name())))
            .collect(Collectors.joining(" "))
    );
    GateInserterMutation wiMutation = new GateInserterMutation(
        new LinkedHashSet<>(ALL_GATES),
        maxNumberOfGates,
        10,
        true
    );
    GateRemoverMutation grMutation = new GateRemoverMutation(10, true);
    WireSwapperMutation wsMutation = new WireSwapperMutation(10, true);
    NetworkCrossover xover = new NetworkCrossover(maxNumberOfGates, XOVER_SUBNET_SIZE_RATE, 10, true);
    // prepare map with stats
    List<Network> all = factory.build(nOfNetworks, rnd);
    SequencedMap<String, List<Network>> map = new LinkedHashMap<>();
    map.put("factory-all", all);
    map.put("factory-good", all.stream().filter(goodGatesPredicate).toList());
    map.put("factory-bad", all.stream().filter(goodGatesPredicate.negate()).toList());
    map.put("gi-mutated-all", all.stream().map(n -> wiMutation.mutate(n, rnd)).toList());
    map.put("gi-mutated-good", map.get("factory-good").stream().map(n -> wiMutation.mutate(n, rnd)).toList());
    map.put("gi-mutated-bad", map.get("factory-bad").stream().map(n -> wiMutation.mutate(n, rnd)).toList());
    map.put("gr-mutated-all", all.stream().map(n -> grMutation.mutate(n, rnd)).toList());
    map.put("gr-mutated-good", map.get("factory-good").stream().map(n -> grMutation.mutate(n, rnd)).toList());
    map.put("gr-mutated-bad", map.get("factory-bad").stream().map(n -> grMutation.mutate(n, rnd)).toList());
    map.put("ws-mutated-all", all.stream().map(n -> wsMutation.mutate(n, rnd)).toList());
    map.put("ws-mutated-good", map.get("factory-good").stream().map(n -> wsMutation.mutate(n, rnd)).toList());
    map.put("ws-mutated-bad", map.get("factory-bad").stream().map(n -> wsMutation.mutate(n, rnd)).toList());
    map.put(
        "xover-all",
        all.stream().map(n -> xover.recombine(n, Misc.pickRandomly(all, rnd), rnd)).toList()
    );
    map.put(
        "xover-good",
        map.get("factory-good")
            .stream()
            .map(n -> xover.recombine(n, Misc.pickRandomly(map.get("factory-good"), rnd), rnd))
            .toList()
    );
    map.put(
        "xover-bad",
        map.get("factory-bad")
            .stream()
            .map(n -> xover.recombine(n, Misc.pickRandomly(map.get("factory-bad"), rnd), rnd))
            .toList()
    );
    // print table
    map.forEach(
        (name, networks) -> System.out.printf("%16.16s (%4d) -> %s%n", name, networks.size(), sStatter.apply(networks))
    );
  }

  public static void main(String[] args) throws ProgramExecutionException {
    factoryStats(
        List.of(Composed.sequence(Base.REAL), Composed.sequence(Base.REAL)),
        List.of(Base.REAL),
        100,
        20,
        10,
        new Random(1)
    );
  }

}
