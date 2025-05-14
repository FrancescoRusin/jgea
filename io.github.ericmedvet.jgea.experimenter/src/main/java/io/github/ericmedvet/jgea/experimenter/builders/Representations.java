/*-
 * ========================LICENSE_START=================================
 * jgea-experimenter
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
package io.github.ericmedvet.jgea.experimenter.builders;

import io.github.ericmedvet.jgea.core.IndependentFactory;
import io.github.ericmedvet.jgea.core.operator.Crossover;
import io.github.ericmedvet.jgea.core.operator.Mutation;
import io.github.ericmedvet.jgea.core.representation.grammar.string.StringGrammar;
import io.github.ericmedvet.jgea.core.representation.grammar.string.cfggp.GrammarBasedSubtreeMutation;
import io.github.ericmedvet.jgea.core.representation.grammar.string.cfggp.GrammarRampedHalfAndHalf;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.*;
import io.github.ericmedvet.jgea.core.representation.sequence.FixedLengthListFactory;
import io.github.ericmedvet.jgea.core.representation.sequence.bit.BitString;
import io.github.ericmedvet.jgea.core.representation.sequence.bit.BitStringFactory;
import io.github.ericmedvet.jgea.core.representation.sequence.bit.BitStringFlipMutation;
import io.github.ericmedvet.jgea.core.representation.sequence.bit.BitStringUniformCrossover;
import io.github.ericmedvet.jgea.core.representation.sequence.integer.IntString;
import io.github.ericmedvet.jgea.core.representation.sequence.integer.IntStringFlipMutation;
import io.github.ericmedvet.jgea.core.representation.sequence.integer.IntStringUniformCrossover;
import io.github.ericmedvet.jgea.core.representation.sequence.integer.UniformIntStringFactory;
import io.github.ericmedvet.jgea.core.representation.sequence.numeric.GaussianMutation;
import io.github.ericmedvet.jgea.core.representation.sequence.numeric.SegmentGeometricCrossover;
import io.github.ericmedvet.jgea.core.representation.sequence.numeric.UniformDoubleFactory;
import io.github.ericmedvet.jgea.core.representation.tree.*;
import io.github.ericmedvet.jgea.core.representation.tree.numeric.Element;
import io.github.ericmedvet.jgea.experimenter.Representation;
import io.github.ericmedvet.jnb.core.Cacheable;
import io.github.ericmedvet.jnb.core.Discoverable;
import io.github.ericmedvet.jnb.core.Param;
import io.github.ericmedvet.jnb.datastructure.Pair;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.SequencedSet;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

@Discoverable(prefixTemplate = "ea.representation|r")
public class Representations {
  private Representations() {
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Function<BitString, Representation<BitString>> bitString(
      @Param(value = "pMutRate", dD = 1d) double pMutRate
  ) {
    return g -> new Representation<>(
        new BitStringFactory(g.size()),
        new BitStringFlipMutation(pMutRate / (double) g.size()),
        Crossover.from(
            new BitStringUniformCrossover()
                .andThen(new BitStringFlipMutation(pMutRate / (double) g.size()))
        )
    );
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <N> Function<Tree<N>, Representation<Tree<N>>> cfgTree(
      @Param("grammar") StringGrammar<N> grammar,
      @Param(value = "minTreeH", dI = 4) int minTreeH,
      @Param(value = "maxTreeH", dI = 16) int maxTreeH
  ) {
    return g -> new Representation<>(
        new GrammarRampedHalfAndHalf<>(minTreeH, maxTreeH, grammar),
        List.of(new GrammarBasedSubtreeMutation<>(maxTreeH, grammar)),
        List.of()
    );
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Function<List<Double>, Representation<List<Double>>> doubleString(
      @Param(value = "initialMinV", dD = -1d) double initialMinV,
      @Param(value = "initialMaxV", dD = 1d) double initialMaxV,
      @Param(value = "sigmaMut", dD = 0.35d) double sigmaMut
  ) {
    return g -> new Representation<>(
        new FixedLengthListFactory<>(g.size(), new UniformDoubleFactory(initialMinV, initialMaxV)),
        new GaussianMutation(sigmaMut),
        Crossover.from(new SegmentGeometricCrossover().andThen(new GaussianMutation(sigmaMut)))
    );
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Function<IntString, Representation<IntString>> intString(
      @Param(value = "pMutRate", dD = 1d) double pMutRate
  ) {
    return g -> new Representation<>(
        new UniformIntStringFactory(g.lowerBound(), g.upperBound(), g.size()),
        new IntStringFlipMutation(pMutRate / (double) g.size()),
        Crossover.from(
            new IntStringUniformCrossover()
                .andThen(new IntStringFlipMutation(pMutRate / (double) g.size()))
        )
    );
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Function<List<Tree<Element>>, Representation<List<Tree<Element>>>> multiSRTree(
      @Param(
          value = "constants", dDs = {0.1, 1, 10}) List<Double> constants,
      @Param(
          value = "operators", dSs = {"addition", "subtraction", "multiplication", "prot_division", "prot_log"}) List<Element.Operator> operators,
      @Param(value = "minTreeH", dI = 4) int minTreeH,
      @Param(value = "maxTreeH", dI = 10) int maxTreeH
  ) {
    return g -> {
      List<Element.Variable> variables = g.stream()
          .map(
              t -> t.visitDepth()
                  .stream()
                  .filter(e -> e instanceof Element.Variable)
                  .map(e -> ((Element.Variable) e).name())
                  .toList()
          )
          .flatMap(List::stream)
          .distinct()
          .map(Element.Variable::new)
          .toList();
      List<Element.Constant> constantElements = constants.stream().map(Element.Constant::new).toList();
      IndependentFactory<Element> terminalFactory = IndependentFactory.oneOf(
          IndependentFactory.picker(variables),
          IndependentFactory.picker(constantElements)
      );
      IndependentFactory<Element> nonTerminalFactory = IndependentFactory.picker(operators);
      IndependentFactory<List<Tree<Element>>> treeListFactory = new FixedLengthListFactory<>(
          g.size(),
          new TreeIndependentFactory<>(minTreeH, maxTreeH, x -> 2, nonTerminalFactory, terminalFactory, 0.5)
      );
      // single tree factory
      TreeBuilder<Element> treeBuilder = new GrowTreeBuilder<>(x -> 2, nonTerminalFactory, terminalFactory);
      // subtree between same position trees
      SubtreeCrossover<Element> subtreeCrossover = new SubtreeCrossover<>(maxTreeH);
      Crossover<List<Tree<Element>>> pairWiseSubtreeCrossover = (list1, list2, rnd) -> IntStream.range(0, list1.size())
          .mapToObj(i -> subtreeCrossover.recombine(list1.get(i), list2.get(i), rnd))
          .toList();
      // swap trees
      Crossover<List<Tree<Element>>> uniformCrossover = (list1, list2, rnd) -> IntStream.range(0, list1.size())
          .mapToObj(i -> rnd.nextDouble() < 0.5 ? list1.get(i) : list2.get(i))
          .toList();
      // subtree mutation
      SubtreeMutation<Element> subtreeMutation = new SubtreeMutation<>(maxTreeH, treeBuilder);
      Mutation<List<Tree<Element>>> allSubtreeMutations = (list, rnd) -> list.stream()
          .map(t -> subtreeMutation.mutate(t, rnd))
          .toList();
      return new Representation<>(
          treeListFactory,
          List.of(allSubtreeMutations),
          List.of(pairWiseSubtreeCrossover, uniformCrossover)
      );
    };
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <G1, G2> Function<Pair<G1, G2>, Representation<Pair<G1, G2>>> pair(
      @Param("first") Function<G1, Representation<G1>> r1,
      @Param("second") Function<G2, Representation<G2>> r2
  ) {
    return p -> Representation.pair(r1.apply(p.first()), r2.apply(p.second()));
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Function<Tree<Element>, Representation<Tree<Element>>> srTree(
      @Param(
          value = "constants", dDs = {0.1, 1, 10}) List<Double> constants,
      @Param(
          value = "operators", dSs = {"addition", "subtraction", "multiplication", "prot_division", "prot_log"}) List<Element.Operator> operators,
      @Param(value = "minTreeH", dI = 4) int minTreeH,
      @Param(value = "maxTreeH", dI = 10) int maxTreeH
  ) {
    return g -> {
      List<Element.Variable> variables = g.visitDepth()
          .stream()
          .filter(e -> e instanceof Element.Variable)
          .map(e -> ((Element.Variable) e).name())
          .distinct()
          .map(Element.Variable::new)
          .toList();
      List<Element.Constant> constantElements = constants.stream().map(Element.Constant::new).toList();
      IndependentFactory<Element> terminalFactory = IndependentFactory.oneOf(
          IndependentFactory.picker(variables),
          IndependentFactory.picker(constantElements)
      );
      IndependentFactory<Element> nonTerminalFactory = IndependentFactory.picker(operators);
      // single tree factory
      TreeBuilder<Element> treeBuilder = new GrowTreeBuilder<>(x -> 2, nonTerminalFactory, terminalFactory);
      return new Representation<>(
          new RampedHalfAndHalf<>(minTreeH, maxTreeH, x -> 2, nonTerminalFactory, terminalFactory),
          new SubtreeMutation<>(maxTreeH, treeBuilder),
          new SubtreeCrossover<>(maxTreeH)
      );
    };
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Function<Network, Representation<Network>> ttpn(
      @Param(value = "maxNOfGates", dI = 32) int maxNOfGates,
      @Param(value = "maxNOfAttempts", dI = 10) int maxNOfAttempts,
      @Param(value = "subnetSizeRate", dD = 0.33) double subnetSizeRate,
      @Param(value = "gates", dNPMs = {"ea.ttpn.gate.bAnd()", "ea.ttpn.gate.bNot()", "ea.ttpn.gate.bOr()", "ea.ttpn" + ".gate.bXor()", "ea.ttpn.gate.concat()", "ea.ttpn.gate.equal()", "ea.ttpn.gate.iTh()", "ea.ttpn" + ".gate.length" + "()", "ea.ttpn.gate.noop()", "ea.ttpn.gate.pairer()", "ea.ttpn.gate.queuer()", "ea.ttpn" + ".gate.select()", "ea" + ".ttpn.gate.sequencer()", "ea.ttpn.gate.sink()", "ea.ttpn.gate" + ".splitter()", "ea" + ".ttpn.gate.unpairer()", "ea" + ".ttpn.gate.iBefore()", "ea.ttpn.gate.iPMathOperator" + "(operator = addition)", "ea.ttpn.gate.iPMathOperator" + "(operator = subtraction)", "ea.ttpn.gate" + ".iPMathOperator(operator = " + "multiplication)", "ea.ttpn.gate" + ".iPMathOperator(operator = division)", "ea.ttpn.gate.iRange()", "ea" + ".ttpn.gate.iSMult()", "ea.ttpn.gate" + ".iSPMult()", "ea.ttpn.gate.iSPSum" + "()", "ea.ttpn.gate.iSSum()", "ea" + ".ttpn.gate.iToR()", "ea.ttpn.gate.rBefore" + "()", "ea.ttpn.gate" + ".repeater()", "ea.ttpn.gate.rPMathOperator" + "(operator = addition)", "ea.ttpn.gate" + ".rPMathOperator" + "(operator = subtraction)", "ea.ttpn.gate" + ".rPMathOperator(operator = multiplication)", "ea" + ".ttpn" + ".gate.rPMathOperator(operator = division)", "ea" + ".ttpn.gate.rSMult()", "ea.ttpn.gate.rSPMult()", "ea" + ".ttpn.gate.rSPSum()", "ea.ttpn.gate.rSSum()", "ea" + ".ttpn.gate.rToI()", "ea.ttpn.gate.sBefore()", "ea.ttpn" + ".gate.sConcat()", "ea.ttpn.gate.sSplitter()", "ea.ttpn.gate.bConst(value = true)", "ea.ttpn" + ".gate.iConst" + "(value = 0)", "ea.ttpn.gate.iConst(value = 1)", "ea.ttpn.gate.iConst(value = 2)", "ea" + ".ttpn.gate.iConst" + "(value = 5)", "ea.ttpn.gate.rConst(value = 0)", "ea.ttpn.gate.rConst(value = 0.1)", "ea.ttpn.gate.rConst" + "(value = 0.2)", "ea.ttpn.gate.rConst(value =" + " 0.5)", "ea.ttpn.gate.sPSequencer()"}) List<Gate> gates,
      @Param("forbiddenGates") List<Gate> forbiddenGates,
      @Param(value = "avoidDeadGates", dB = true) boolean avoidDeadGates
  ) {
    SequencedSet<Gate> actualGates = gates.stream()
        .filter(g -> !forbiddenGates.contains(g))
        .collect(Collectors.toCollection(LinkedHashSet::new));
    return ttpn -> new Representation<>(
        new BackTracingNetworkFactory(
            ttpn.inputGates().values().stream().toList(),
            ttpn.outputGates().values().stream().toList(),
            actualGates,
            maxNOfGates,
            100 * maxNOfAttempts
        ),
        List.of(
            new GateInserterMutation(actualGates, maxNOfGates, maxNOfAttempts, avoidDeadGates),
            new GateRemoverMutation(maxNOfAttempts, avoidDeadGates),
            new WireSwapperMutation(maxNOfAttempts, avoidDeadGates)
        ),
        List.of(new NetworkCrossover(maxNOfGates, subnetSizeRate, maxNOfAttempts, avoidDeadGates))
    );
  }
}
