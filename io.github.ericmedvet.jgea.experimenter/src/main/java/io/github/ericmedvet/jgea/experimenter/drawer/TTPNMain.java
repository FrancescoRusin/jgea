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
package io.github.ericmedvet.jgea.experimenter.drawer;

import io.github.ericmedvet.jgea.core.order.ParetoDominance;
import io.github.ericmedvet.jgea.core.order.PartialComparator;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.InstrumentedProgram;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.Program;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.ProgramExecutionException;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.*;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.Base;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.Composed;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.StringParser;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.TypeException;
import io.github.ericmedvet.jgea.core.representation.tree.numeric.Element;
import io.github.ericmedvet.jgea.core.util.IntRange;
import io.github.ericmedvet.jgea.problem.programsynthesis.Problems;
import io.github.ericmedvet.jgea.problem.programsynthesis.ProgramSynthesisProblem;
import io.github.ericmedvet.jgea.problem.programsynthesis.synthetic.PrecomputedSyntheticPSProblem;
import io.github.ericmedvet.jnb.core.NamedBuilder;
import io.github.ericmedvet.jnb.datastructure.DoubleRange;
import java.io.File;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.random.RandomGenerator;

public class TTPNMain {

  private static void comparator() {
    ParetoDominance<Double> basePC = new ParetoDominance<>(
        List.of(
            Double::compareTo,
            Double::compareTo
        )
    );
    PartialComparator<List<Double>> extendedBasePC = basePC.on(vs -> vs.subList(0, 2));
    PartialComparator<List<Double>> addedPC = PartialComparator.from(Double::compareTo).on(List::getLast);
    System.out.println(basePC.compare(List.of(1d, 2d), List.of(3d, 4d)));
    System.out.println(basePC.compare(List.of(1d, 5d), List.of(3d, 4d)));
    System.out.println(addedPC.compare(List.of(2d, 1d, 4d), List.of(4d, 3d, 4d)));
    System.out.println(extendedBasePC.compare(List.of(2d, 1d, 4d), List.of(4d, 3d, 4d)));
    System.out.println(
        ParetoDominance.compare(List.of(2d, 1d, 4d), List.of(4d, 3d, 4d), List.of(extendedBasePC, addedPC))
    );
  }

  private static void doComputationStuff() throws NoSuchMethodException, ProgramExecutionException, NetworkStructureException, TypeException {
    Network n = new Network(
        List.of(
            Gate.input(Composed.sequence(Base.REAL)),
            Gate.input(Composed.sequence(Base.REAL)),
            Gates.splitter(),
            Gates.splitter(),
            Gates.rPMathOperator(Element.Operator.MULTIPLICATION),
            Gates.rSPSum(),
            Gate.output(Base.REAL)
        ),
        Set.of(
            Wire.of(0, 0, 2, 0),
            Wire.of(1, 0, 3, 0),
            Wire.of(2, 0, 4, 0),
            Wire.of(3, 0, 4, 1),
            Wire.of(4, 0, 5, 0),
            Wire.of(5, 0, 5, 1),
            Wire.of(5, 0, 6, 0)
        )
    );
    DataFactory df = new DataFactory(
        List.of(1, 2, 3),
        List.of(1d, 2d, 3d, 1.5, 2.5, 3.14),
        List.of("cat", "dog", "Hello World!", "mummy"),
        new IntRange(-10, 100),
        new DoubleRange(-10, 10),
        new IntRange(2, 20),
        new IntRange(3, 8)
    );
    RandomGenerator rnd = new Random(1);
    System.out.println(df.apply(StringParser.parse("[<S,[I],R>]"), rnd));

    Program tProgram = Program.from(Problems.class.getMethod("vProduct", List.class, List.class));
    System.out.println(tProgram);
    InstrumentedProgram ttpnProgram = new Runner(1000, 1000, 1000, 100, false).asInstrumentedProgram(n);
    List<Object> inputs = List.of(List.of(1d, 2d), List.of(3d, 4d));
    System.out.println(tProgram.run(inputs));
    InstrumentedProgram.Outcome o = ttpnProgram.runInstrumented(inputs);
    System.out.println(o);

    ProgramSynthesisProblem psp = new PrecomputedSyntheticPSProblem(
        tProgram,
        List.of(ProgramSynthesisProblem.Metric.FAIL_RATE, ProgramSynthesisProblem.Metric.AVG_RAW_DISSIMILARITY),
        100d,
        df,
        rnd,
        10,
        5,
        0.5
    );

    //System.out.println(psp.qualityFunction().apply(tProgram));
    System.out.println(psp.qualityFunction().apply(ttpnProgram));
    System.out.println(psp.validationQualityFunction().apply(ttpnProgram));

    psp.caseProvider()
        .stream()
        .forEach(
            example -> System.out.printf(
                "\tactual=%s vs. predicted=%s%n",
                example.output().outputs(),
                psp.predictFunction().apply(ttpnProgram, example.input()).outputs()
            )
        );
  }

  private static void doFactoryStuff() throws NetworkStructureException, TypeException {
    Network sn = new Network(
        List.of(
            Gate.input(Composed.sequence(Base.STRING)),
            Gates.splitter(),
            Gate.output(Base.STRING)
        ),
        Set.of(
            Wire.of(0, 0, 1, 0),
            Wire.of(1, 0, 2, 0)
        )
    );
    System.out.println(sn);

    Network n = new Network(
        List.of(
            Gate.input(Composed.sequence(Base.REAL)),
            Gate.input(Composed.sequence(Base.REAL)),
            Gates.splitter(),
            Gates.splitter(),
            Gates.rPMathOperator(Element.Operator.MULTIPLICATION),
            Gates.rSPSum(),
            Gate.output(Base.REAL)
            //new ones
            /*Gates.rSPSum(),
            Gates.rSPSum(),
            Gates.rSPSum(),
            Gate.output(Base.REAL)*/
        ),
        Set.of(
            Wire.of(0, 0, 2, 0),
            Wire.of(1, 0, 3, 0),
            Wire.of(2, 0, 4, 0),
            Wire.of(3, 0, 4, 1),
            Wire.of(4, 0, 5, 0),
            Wire.of(5, 0, 5, 1),
            Wire.of(5, 0, 6, 0)
            //new ones
            /*Wire.of(2, 0, 7, 0),
            Wire.of(9, 0, 7, 1),
            Wire.of(7, 0, 8, 0),
            Wire.of(9, 0, 8, 1),
            Wire.of(8, 0, 9, 0),
            Wire.of(8, 0, 9, 1),
            Wire.of(9, 0, 10, 0)*/
        )
    );
    System.out.println("===\n" + n);

    Network pn = new Network(
        List.of(
            Gates.rPMathOperator(Element.Operator.MULTIPLICATION),
            Gates.rPMathOperator(Element.Operator.ADDITION)
        ),
        Set.of(
            Wire.of(0, 0, 1, 1),
            Wire.of(1, 0, 0, 1)
        )
    );
    System.out.println("===\n" + pn);

    RandomGenerator rnd = new Random();
    TTPNDrawer drawer = new TTPNDrawer(TTPNDrawer.Configuration.DEFAULT);
    //drawer.show(n);
    //drawer.show(pn);
    //drawer.show(mn);
    //drawer.show(new ImageBuilder.ImageInfo(600, 300), n);

    BackTracingNetworkFactory nf = new BackTracingNetworkFactory(
        List.of(Composed.sequence(Base.REAL), Composed.sequence(Base.REAL)),
        List.of(Base.REAL),
        new LinkedHashSet<>(StatsMain.ALL_GATES),
        20,
        10
    );
    Network newN = nf.build(rnd);
    //drawer.show(newN);
    //drawer.show(NetworkUtils.randomSubnetwork(newN, rnd, newN.gates().size() / 4));

    for (int i = 0; i < 1000; i++) {
      Network nn = nf.build(rnd);
      Network snn = NetworkUtils.randomSubnetwork(newN, rnd, nn.gates().size() / 2);
      Network hnn = NetworkUtils.randomHoledNetwork(newN, rnd, nn.gates().size() / 4);
      System.out.printf(
          "n.g:%3d n.w:%3d\tsn.g:%3d sn.w:%3d\thn.g:%3d hn.w:%3d\tsubnets:%2d%n",
          nn.gates().size(),
          nn.wires().size(),
          snn.gates().size(),
          snn.wires().size(),
          hnn.gates().size(),
          hnn.wires().size(),
          hnn.disjointSubnetworks().size()
      );
    }

  }

  private static void factory() {
    RandomGenerator rnd = new Random();
    BackTracingNetworkFactory factory = new BackTracingNetworkFactory(
        List.of(Composed.sequence(Base.REAL), Composed.sequence(Base.REAL)),
        List.of(Base.REAL),
        new LinkedHashSet<>(StatsMain.ALL_GATES),
        16,
        1000
    );
    TTPNDrawer drawer = new TTPNDrawer(TTPNDrawer.Configuration.DEFAULT);
    Network network = factory.build(rnd);
    drawer.show(network);
    System.out.println(network);

    //IntStream.range(0, 1000).forEach(i -> factory.build(rnd, n -> System.out.printf("======%n%s%n===%n", n)));
  }

  private static void iArraySum() throws NoSuchMethodException, NetworkStructureException, TypeException, ProgramExecutionException {
    Program target = Program.from(Problems.class.getMethod("iArraySum", List.class));
    Network n = new Network(
        List.of(
            Gate.input(Composed.sequence(Base.INT)),
            Gates.splitter(),
            Gates.iToR(),
            Gates.rSPSum(),
            Gates.rToI(),
            Gate.output(Base.INT),
            Gates.iConst(5)
        ),
        Set.of(
            Wire.of(0, 0, 1, 0),
            Wire.of(1, 0, 2, 0),
            Wire.of(2, 0, 3, 0),
            Wire.of(3, 0, 4, 0),
            Wire.of(3, 0, 3, 1),
            Wire.of(4, 0, 5, 0),
            Wire.of(1, 0, 6, 0)
        )
    );
    TTPNDrawer drawer = new TTPNDrawer(TTPNDrawer.Configuration.DEFAULT);
    //drawer.show(n);
    Runner runner = new Runner(1000, 1000, 1000, 100, false);
    Runner.TTPNInstrumentedOutcome outcome = runner.run(n, List.of(List.of(1, 2, 3, 4, 5, 6, 7, 8)));
    TTPNOutcomeVideoBuilder videoBuilder = new TTPNOutcomeVideoBuilder(TTPNOutcomeVideoBuilder.Configuration.DEFAULT);
    videoBuilder.save(new File("../ttpn-iArrayMax.mp4"), outcome);
    System.out.println(outcome);
  }

  private static void iBiMax() throws NetworkStructureException, TypeException {
    NamedBuilder<?> nb = NamedBuilder.fromDiscovery();
    ProgramSynthesisProblem psb = (ProgramSynthesisProblem) nb.build("ea.p.ps.synthetic(name = \"iBiMax\")");
    // good solution
    Network goodNetwork = new Network(
        List.of(
            Gate.input(Base.INT),
            Gate.input(Base.INT),
            Gate.output(Base.INT),
            Gates.iBefore(),
            Gates.select()
        ),
        Set.of(
            Wire.of(0, 0, 3, 0),
            Wire.of(1, 0, 3, 1),
            Wire.of(0, 0, 4, 1),
            Wire.of(1, 0, 4, 0),
            Wire.of(3, 0, 4, 2),
            Wire.of(4, 0, 2, 0)
        )
    );
    TTPNDrawer drawer = new TTPNDrawer(TTPNDrawer.Configuration.DEFAULT);
    //drawer.show(goodNetwork);
    Runner runner = new Runner(100, 100, 100, 100, false);
    // check
    psb.caseProvider()
        .stream()
        .forEach(
            e -> System.out.printf(
                "in=%s\tactualOut=%s\tpredOut=%s%n",
                e.input(),
                e.output().outputs(),
                runner.run(goodNetwork, e.input()).outputs()
            )
        );
    Runner.TTPNInstrumentedOutcome outcome = runner.run(goodNetwork, List.of(3, 4));
    System.out.println(outcome);
    TTPNOutcomeVideoBuilder videoBuilder = new TTPNOutcomeVideoBuilder(TTPNOutcomeVideoBuilder.Configuration.DEFAULT);
    videoBuilder.save(new File("../ttpn-iBiMax.mp4"), outcome);
  }

  private static void loopedNet() throws NetworkStructureException, TypeException {
    Network n = new Network(
        List.of(
            Gate.input(Base.INT),
            Gates.pairer()
        ),
        Set.of(
          //Wire.of(0,0,1,0),
          //Wire.of(1,0,1,1)
        )
    );
    TTPNDrawer drawer = new TTPNDrawer(TTPNDrawer.Configuration.DEFAULT);
    drawer.show(n);
    drawer.show(n.wireFreeInputEndPoints(ts -> 0));
  }

  public static void main(
      String[] args
  ) throws NetworkStructureException, ProgramExecutionException, NoSuchMethodException, TypeException {
    //weirdOne();
    //factory();
    //doComputationStuff();
    //comparator();
    //xover();
    iArraySum();
    //iBiMax();
  }

  private static void weirdOne() throws NetworkStructureException, TypeException {
    Network n = new Network(
        List.of(
            Gate.input(Composed.sequence(Base.REAL)),
            Gates.select(),
            Gates.length()
        ),
        Set.of(
            Wire.of(0, 0, 1, 1),
            Wire.of(1, 0, 2, 0)
        )
    );
    new TTPNDrawer(TTPNDrawer.Configuration.DEFAULT).show(n);
  }

  private static void xover() {
    RandomGenerator rnd = new Random(1);
    BackTracingNetworkFactory factory = new BackTracingNetworkFactory(
        List.of(Composed.sequence(Base.REAL), Composed.sequence(Base.REAL)),
        List.of(Base.REAL),
        new LinkedHashSet<>(StatsMain.ALL_GATES),
        16,
        1000
    );
    TTPNDrawer drawer = new TTPNDrawer(TTPNDrawer.Configuration.DEFAULT);
    Network n1 = factory.build(rnd);
    Network n2 = factory.build(rnd);
    System.out.println(n1);
    System.out.println("===");
    System.out.println(n2);
    Network xovered = new NetworkCrossover(10, .5, 10, true).recombine(n1, n2, rnd);
    drawer.show(n1);
    drawer.show(n2);
    System.out.println("===");
    System.out.println(xovered);
    drawer.show(xovered);
  }

}
