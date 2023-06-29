/*
 * Copyright 2023 eric
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.ericmedvet.jgea.experimenter.builders;

import io.github.ericmedvet.jgea.core.Factory;
import io.github.ericmedvet.jgea.core.IndependentFactory;
import io.github.ericmedvet.jgea.core.distance.Jaccard;
import io.github.ericmedvet.jgea.core.operator.Crossover;
import io.github.ericmedvet.jgea.core.operator.GeneticOperator;
import io.github.ericmedvet.jgea.core.operator.Mutation;
import io.github.ericmedvet.jgea.core.problem.QualityBasedProblem;
import io.github.ericmedvet.jgea.core.representation.graph.*;
import io.github.ericmedvet.jgea.core.representation.graph.numeric.Constant;
import io.github.ericmedvet.jgea.core.representation.graph.numeric.Input;
import io.github.ericmedvet.jgea.core.representation.graph.numeric.Output;
import io.github.ericmedvet.jgea.core.representation.graph.numeric.operatorgraph.BaseOperator;
import io.github.ericmedvet.jgea.core.representation.graph.numeric.operatorgraph.OperatorGraph;
import io.github.ericmedvet.jgea.core.representation.graph.numeric.operatorgraph.OperatorNode;
import io.github.ericmedvet.jgea.core.representation.graph.numeric.operatorgraph.ShallowFactory;
import io.github.ericmedvet.jgea.core.representation.sequence.FixedLengthListFactory;
import io.github.ericmedvet.jgea.core.representation.sequence.UniformCrossover;
import io.github.ericmedvet.jgea.core.representation.sequence.integer.IntFlipMutation;
import io.github.ericmedvet.jgea.core.representation.sequence.integer.IntString;
import io.github.ericmedvet.jgea.core.representation.sequence.integer.UniformIntStringFactory;
import io.github.ericmedvet.jgea.core.representation.sequence.numeric.GaussianMutation;
import io.github.ericmedvet.jgea.core.representation.sequence.numeric.UniformDoubleFactory;
import io.github.ericmedvet.jgea.core.representation.tree.*;
import io.github.ericmedvet.jgea.core.representation.tree.numeric.Element;
import io.github.ericmedvet.jgea.core.selector.Last;
import io.github.ericmedvet.jgea.core.selector.Tournament;
import io.github.ericmedvet.jgea.core.solver.*;
import io.github.ericmedvet.jgea.core.solver.speciation.LazySpeciator;
import io.github.ericmedvet.jgea.core.solver.speciation.SpeciatedEvolver;
import io.github.ericmedvet.jgea.core.solver.state.POSetPopulationState;
import io.github.ericmedvet.jgea.experimenter.InvertibleMapper;
import io.github.ericmedvet.jnb.core.Param;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * @author "Eric Medvet" on 2022/11/21 for 2d-robot-evolution
 */
public class Solvers {

  private Solvers() {
  }

  @SuppressWarnings("unused")
  public static <S, Q> Function<S, StandardEvolver<POSetPopulationState<IntString, S, Q>, QualityBasedProblem<S, Q>,
      IntString, S, Q>> intGA(
      @Param(value = "mapper") InvertibleMapper<IntString, S> mapper,
      @Param(value = "crossoverP", dD = 0.8d) double crossoverP,
      @Param(value = "pMut", dD = 0.01d) double pMut,
      @Param(value = "tournamentRate", dD = 0.05d) double tournamentRate,
      @Param(value = "minNTournament", dI = 3) int minNTournament,
      @Param(value = "nPop", dI = 100) int nPop,
      @Param(value = "nEval") int nEval,
      @Param(value = "diversity", dB = true) boolean diversity,
      @Param(value = "remap") boolean remap
  ) {
    return exampleS -> {
      IntString exampleGenotype = mapper.exampleFor(exampleS);
      IndependentFactory<IntString> factory = new UniformIntStringFactory(
          exampleGenotype.getLowerBound(),
          exampleGenotype.getUpperBound(),
          exampleGenotype.size()
      );
      Map<GeneticOperator<IntString>, Double> geneticOperators = Map.ofEntries(
          Map.entry(new IntFlipMutation(pMut), 1d - crossoverP),
          Map.entry(new UniformCrossover<>(factory).andThen(new IntFlipMutation(pMut)), crossoverP)
      );
      if (!diversity) {
        return new StandardEvolver<>(
            mapper.mapperFor(exampleS),
            factory,
            nPop,
            StopConditions.nOfFitnessEvaluations(nEval),
            geneticOperators,
            new Tournament(Math.max(minNTournament, (int) Math.ceil((double) nPop * tournamentRate))),
            new Last(),
            nPop,
            true,
            remap,
            (p, r) -> new POSetPopulationState<>()
        );
      } else {
        return new StandardWithEnforcedDiversityEvolver<>(
            mapper.mapperFor(exampleS),
            factory,
            nPop,
            StopConditions.nOfFitnessEvaluations(nEval),
            geneticOperators,
            new Tournament(Math.max(minNTournament, (int) Math.ceil((double) nPop * tournamentRate))),
            new Last(),
            nPop,
            true,
            remap,
            (p, r) -> new POSetPopulationState<>(),
            100
        );
      }
    };
  }

  @SuppressWarnings("unused")
  public static <S, Q> Function<S, StandardEvolver<POSetPopulationState<List<Tree<Element>>, S, Q>,
      QualityBasedProblem<S, Q>,
      List<Tree<Element>>, S, Q>> multiSRTreeGP(
      @Param(value = "mapper") InvertibleMapper<List<Tree<Element>>, S> mapper,
      @Param(value = "minConst", dD = 0d) double minConst,
      @Param(value = "maxConst", dD = 5d) double maxConst,
      @Param(value = "nConst", dI = 10) int nConst,
      @Param(value = "operators", dSs = {
          "addition",
          "subtraction",
          "multiplication",
          "prot_division",
          "prot_log"
      }) List<Element.Operator> operators,
      @Param(value = "minTreeH", dI = 3) int minTreeH,
      @Param(value = "maxTreeH", dI = 8) int maxTreeH,
      @Param(value = "crossoverP", dD = 0.8d) double crossoverP,
      @Param(value = "tournamentRate", dD = 0.05d) double tournamentRate,
      @Param(value = "minNTournament", dI = 3) int minNTournament,
      @Param(value = "nPop", dI = 100) int nPop,
      @Param(value = "nEval") int nEval,
      @Param(value = "diversity", dB = true) boolean diversity,
      @Param(value = "nAttemptsDiversity", dI = 100) int nAttemptsDiversity,
      @Param(value = "remap") boolean remap
  ) {
    return exampleS -> {
      List<Element.Variable> variables = mapper.exampleFor(exampleS).stream().map(t -> t.visitDepth()
          .stream()
          .filter(e -> e instanceof Element.Variable)
          .map(e -> ((Element.Variable) e).name())
          .toList()).flatMap(List::stream).distinct().map(Element.Variable::new).toList();
      double constStep = (maxConst - minConst) / nConst;
      List<Element.Constant> constants = DoubleStream.iterate(minConst, d -> d + constStep)
          .limit(nConst)
          .mapToObj(Element.Constant::new)
          .toList();
      IndependentFactory<Element> terminalFactory = IndependentFactory.oneOf(
          IndependentFactory.picker(variables),
          IndependentFactory.picker(constants)
      );
      IndependentFactory<Element> nonTerminalFactory = IndependentFactory.picker(operators);
      IndependentFactory<List<Tree<Element>>> treeListFactory = new FixedLengthListFactory<>(
          mapper.exampleFor(exampleS).size(),
          new RampedHalfAndHalf<>(minTreeH, maxTreeH, x -> 2, nonTerminalFactory, terminalFactory).independent()
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
      Mutation<List<Tree<Element>>> allSubtreeMutations = (list, rnd) -> list.stream().map(t -> subtreeMutation.mutate(
          t,
          rnd
      )).toList();
      Map<GeneticOperator<List<Tree<Element>>>, Double> geneticOperators = Map.ofEntries(
          Map.entry(
              pairWiseSubtreeCrossover,
              crossoverP / 2d
          ),
          Map.entry(uniformCrossover, crossoverP / 2d),
          Map.entry(allSubtreeMutations, 1d - crossoverP)
      );
      if (!diversity) {
        return new StandardEvolver<>(
            mapper.mapperFor(exampleS),
            treeListFactory,
            nPop,
            StopConditions.nOfFitnessEvaluations(nEval),
            geneticOperators,
            new Tournament(Math.max(minNTournament, (int) Math.ceil((double) nPop * tournamentRate))),
            new Last(),
            nPop,
            true,
            remap,
            (p, r) -> new POSetPopulationState<>()
        );
      }
      return new StandardWithEnforcedDiversityEvolver<>(
          mapper.mapperFor(exampleS),
          treeListFactory,
          nPop,
          StopConditions.nOfFitnessEvaluations(nEval),
          geneticOperators,
          new Tournament(Math.max(minNTournament, (int) Math.ceil((double) nPop * tournamentRate))),
          new Last(),
          nPop,
          true,
          remap,
          (p, r) -> new POSetPopulationState<>(),
          nAttemptsDiversity
      );
    };
  }

  @SuppressWarnings("unused")
  public static <S, Q> Function<S, StandardEvolver<POSetPopulationState<List<Double>, S, Q>, QualityBasedProblem<S, Q>,
      List<Double>, S, Q>> numGA(
      @Param(value = "mapper") InvertibleMapper<List<Double>, S> mapper,
      @Param(value = "initialMinV", dD = -1d) double initialMinV,
      @Param(value = "initialMaxV", dD = 1d) double initialMaxV,
      @Param(value = "crossoverP", dD = 0.8d) double crossoverP,
      @Param(value = "sigmaMut", dD = 0.35d) double sigmaMut,
      @Param(value = "tournamentRate", dD = 0.05d) double tournamentRate,
      @Param(value = "minNTournament", dI = 3) int minNTournament,
      @Param(value = "nPop", dI = 100) int nPop,
      @Param(value = "nEval") int nEval,
      @Param(value = "diversity") boolean diversity,
      @Param(value = "remap") boolean remap
  ) {
    return exampleS -> {
      IndependentFactory<List<Double>> doublesFactory = new FixedLengthListFactory<>(
          mapper.exampleFor(exampleS).size(),
          new UniformDoubleFactory(initialMinV, initialMaxV)
      );
      Map<GeneticOperator<List<Double>>, Double> geneticOperators = Map.ofEntries(
          Map.entry(new GaussianMutation(sigmaMut), 1d - crossoverP),
          Map.entry(new UniformCrossover<>(doublesFactory).andThen(new GaussianMutation(sigmaMut)), crossoverP)
      );
      if (!diversity) {
        return new StandardEvolver<>(
            mapper.mapperFor(exampleS),
            doublesFactory,
            nPop,
            StopConditions.nOfFitnessEvaluations(nEval),
            geneticOperators,
            new Tournament(Math.max(minNTournament, (int) Math.ceil((double) nPop * tournamentRate))),
            new Last(),
            nPop,
            true,
            remap,
            (p, r) -> new POSetPopulationState<>()
        );
      } else {
        return new StandardWithEnforcedDiversityEvolver<>(
            mapper.mapperFor(exampleS),
            doublesFactory,
            nPop,
            StopConditions.nOfFitnessEvaluations(nEval),
            geneticOperators,
            new Tournament(Math.max(minNTournament, (int) Math.ceil((double) nPop * tournamentRate))),
            new Last(),
            nPop,
            true,
            remap,
            (p, r) -> new POSetPopulationState<>(),
            100
        );
      }
    };
  }

  @SuppressWarnings("unused")
  public static <S, Q> Function<S, SpeciatedEvolver<QualityBasedProblem<S, Q>, Graph<Node,
      OperatorGraph.NonValuedArc>, S, Q>> oGraphea(
      @Param(value = "mapper") InvertibleMapper<Graph<Node, OperatorGraph.NonValuedArc>, S> mapper,
      @Param(value = "minConst", dD = 0d) double minConst,
      @Param(value = "maxConst", dD = 5d) double maxConst,
      @Param(value = "nConst", dI = 10) int nConst,
      @Param(value = "operators", dSs = {"addition", "subtraction", "multiplication", "prot_division", "prot_log"}) List<BaseOperator> operators,
      @Param(value = "nPop", dI = 100) int nPop,
      @Param(value = "nEval") int nEval,
      @Param(value = "arcAdditionRate", dD = 3d) double arcAdditionRate,
      @Param(value = "arcRemovalRate", dD = 0.1d) double arcRemovalRate,
      @Param(value = "nodeAdditionRate", dD = 1d) double nodeAdditionRate,
      @Param(value = "nPop", dI = 5) int minSpeciesSizeForElitism,
      @Param(value = "rankBase", dD = 0.75d) double rankBase,
      @Param(value = "remap") boolean remap
  ) {
    return exampleS -> {
      Map<GeneticOperator<Graph<Node, OperatorGraph.NonValuedArc>>, Double> geneticOperators =
          Map.ofEntries(
              Map.entry(new NodeAddition<Node, OperatorGraph.NonValuedArc>(
                  OperatorNode.sequentialIndexFactory(operators.toArray(BaseOperator[]::new)),
                  Mutation.copy(),
                  Mutation.copy()
              ).withChecker(OperatorGraph.checker()), nodeAdditionRate),
              Map.entry(new ArcAddition<Node, OperatorGraph.NonValuedArc>(
                  r -> OperatorGraph.NON_VALUED_ARC,
                  false
              ).withChecker(OperatorGraph.checker()), arcAdditionRate),
              Map.entry(new ArcRemoval<Node, OperatorGraph.NonValuedArc>(node -> (node instanceof Input) || (node instanceof Constant) || (node instanceof Output)).withChecker(
                  OperatorGraph.checker()), arcRemovalRate)
          );
      Graph<Node, OperatorGraph.NonValuedArc> graph = mapper.exampleFor(exampleS);
      double constStep = (maxConst - minConst) / nConst;
      List<Double> constants = DoubleStream.iterate(minConst, d -> d + constStep).limit(nConst).boxed().toList();
      return new SpeciatedEvolver<>(
          mapper.mapperFor(exampleS),
          new ShallowFactory(
              graph.nodes()
                  .stream()
                  .filter(n -> n instanceof Input)
                  .map(n -> ((Input) n).getName())
                  .distinct()
                  .toList(),
              graph.nodes()
                  .stream()
                  .filter(n -> n instanceof Output)
                  .map(n -> ((Output) n).getName())
                  .distinct()
                  .toList(),
              constants
          ),
          nPop,
          StopConditions.nOfFitnessEvaluations(nEval),
          geneticOperators,
          remap,
          minSpeciesSizeForElitism,
          new LazySpeciator<>((new Jaccard()).on(i -> i.genotype().nodes()), 0.25),
          rankBase
      );
    };
  }

  @SuppressWarnings("unused")
  public static <S, Q> Function<S, OpenAIEvolutionaryStrategy<S, Q>> openAIES(
      @Param(value = "mapper") InvertibleMapper<List<Double>, S> mapper,
      @Param(value = "initialMinV", dD = -1d) double initialMinV,
      @Param(value = "initialMaxV", dD = 1d) double initialMaxV,
      @Param(value = "sigma", dD = 0.35d) double sigma,
      @Param(value = "batchSize", dI = 15) int batchSize,
      @Param(value = "nEval") int nEval
  ) {
    return exampleS -> new OpenAIEvolutionaryStrategy<>(
        mapper.mapperFor(exampleS),
        new FixedLengthListFactory<>(
            mapper.exampleFor(exampleS).size(),
            new UniformDoubleFactory(initialMinV, initialMaxV)
        ),
        batchSize,
        StopConditions.nOfFitnessEvaluations(nEval),
        sigma
    );
  }

  @SuppressWarnings("unused")
  public static <S, Q> Function<S, SimpleEvolutionaryStrategy<S, Q>> simpleES(
      @Param(value = "mapper") InvertibleMapper<List<Double>, S> mapper,
      @Param(value = "initialMinV", dD = -1d) double initialMinV,
      @Param(value = "initialMaxV", dD = 1d) double initialMaxV,
      @Param(value = "sigma", dD = 0.35d) double sigma,
      @Param(value = "parentsRate", dD = 0.33d) double parentsRate,
      @Param(value = "nOfElites", dI = 1) int nOfElites,
      @Param(value = "nPop", dI = 30) int nPop,
      @Param(value = "nEval") int nEval,
      @Param(value = "remap") boolean remap
  ) {
    return exampleS -> new SimpleEvolutionaryStrategy<>(
        mapper.mapperFor(exampleS),
        new FixedLengthListFactory<>(
            mapper.exampleFor(exampleS).size(),
            new UniformDoubleFactory(initialMinV, initialMaxV)
        ),
        nPop,
        StopConditions.nOfFitnessEvaluations(nEval),
        nOfElites,
        (int) Math.round(nPop * parentsRate),
        sigma,
        remap
    );
  }

  @SuppressWarnings("unused")
  public static <S, Q> Function<S, StandardEvolver<POSetPopulationState<Tree<Element>, S, Q>, QualityBasedProblem<S, Q>,
      Tree<Element>, S, Q>> srTreeGP(
      @Param(value = "mapper") InvertibleMapper<Tree<Element>, S> mapper,
      @Param(value = "minConst", dD = 0d) double minConst,
      @Param(value = "maxConst", dD = 5d) double maxConst,
      @Param(value = "nConst", dI = 10) int nConst,
      @Param(value = "operators", dSs = {
          "addition",
          "subtraction",
          "multiplication",
          "prot_division",
          "prot_log"
      }) List<Element.Operator> operators,
      @Param(value = "minTreeH", dI = 3) int minTreeH,
      @Param(value = "maxTreeH", dI = 8) int maxTreeH,
      @Param(value = "crossoverP", dD = 0.8d) double crossoverP,
      @Param(value = "tournamentRate", dD = 0.05d) double tournamentRate,
      @Param(value = "minNTournament", dI = 3) int minNTournament,
      @Param(value = "nPop", dI = 100) int nPop,
      @Param(value = "nEval") int nEval,
      @Param(value = "diversity", dB = true) boolean diversity,
      @Param(value = "nAttemptsDiversity", dI = 100) int nAttemptsDiversity,
      @Param(value = "remap") boolean remap
  ) {
    return exampleS -> {
      List<Element.Variable> variables = mapper.exampleFor(exampleS)
          .visitDepth()
          .stream()
          .filter(e -> e instanceof Element.Variable)
          .map(e -> ((Element.Variable) e).name())
          .distinct()
          .map(Element.Variable::new)
          .toList();
      double constStep = (maxConst - minConst) / nConst;
      List<Element.Constant> constants = DoubleStream.iterate(minConst, d -> d + constStep)
          .limit(nConst)
          .mapToObj(Element.Constant::new)
          .toList();
      IndependentFactory<Element> terminalFactory = IndependentFactory.oneOf(
          IndependentFactory.picker(variables),
          IndependentFactory.picker(constants)
      );
      IndependentFactory<Element> nonTerminalFactory = IndependentFactory.picker(operators);
      // single tree factory
      TreeBuilder<Element> treeBuilder = new GrowTreeBuilder<>(x -> 2, nonTerminalFactory, terminalFactory);
      Factory<Tree<Element>> treeFactory = new RampedHalfAndHalf<>(
          minTreeH,
          maxTreeH,
          x -> 2,
          nonTerminalFactory,
          terminalFactory
      );
      // operators
      Map<GeneticOperator<Tree<Element>>, Double> geneticOperators = Map.ofEntries(
          Map.entry(new SubtreeCrossover<>(maxTreeH), crossoverP),
          Map.entry(new SubtreeMutation<>(maxTreeH, treeBuilder), 1d - crossoverP)
      );
      if (!diversity) {
        return new StandardEvolver<>(
            mapper.mapperFor(exampleS),
            treeFactory,
            nPop,
            StopConditions.nOfFitnessEvaluations(nEval),
            geneticOperators,
            new Tournament(Math.max(minNTournament, (int) Math.ceil((double) nPop * tournamentRate))),
            new Last(),
            nPop,
            true,
            remap,
            (p, r) -> new POSetPopulationState<>()
        );
      }
      return new StandardWithEnforcedDiversityEvolver<>(
          mapper.mapperFor(exampleS),
          treeFactory,
          nPop,
          StopConditions.nOfFitnessEvaluations(nEval),
          geneticOperators,
          new Tournament(Math.max(minNTournament, (int) Math.ceil((double) nPop * tournamentRate))),
          new Last(),
          nPop,
          true,
          remap,
          (p, r) -> new POSetPopulationState<>(),
          nAttemptsDiversity
      );
    };
  }

}
