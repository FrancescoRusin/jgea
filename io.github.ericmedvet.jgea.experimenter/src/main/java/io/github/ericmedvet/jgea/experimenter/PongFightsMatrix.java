/*-
 * ========================LICENSE_START=================================
 * jgea-experimenter
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
package io.github.ericmedvet.jgea.experimenter;

import io.github.ericmedvet.jgea.core.InvertibleMapper;
import io.github.ericmedvet.jnb.core.NamedBuilder;
import io.github.ericmedvet.jnb.datastructure.DoubleRange;
import io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction;
import io.github.ericmedvet.jnb.datastructure.Pair;
import io.github.ericmedvet.jsdynsym.control.BiSimulation;
import io.github.ericmedvet.jsdynsym.control.HomogeneousBiAgentTask;
import io.github.ericmedvet.jsdynsym.control.pong.PongAgent;
import io.github.ericmedvet.jsdynsym.control.pong.PongEnvironment;
import io.github.ericmedvet.jsdynsym.core.numerical.NumericalDynamicalSystem;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

public class PongFightsMatrix {
  private static final NamedBuilder<Object> BUILDER = NamedBuilder.fromDiscovery();

  @SuppressWarnings("unchecked")
  public static void main(String[] args) throws IOException {

    String folder = "/home/il_bello/IdeaProjects/results/pong-ablation-study/2025-04-07--14-56-10/";
    String CSVPath1 = folder + "allBest.csv";
    boolean singleCSV = true;
    String CSVPath2 = folder + "";
    String delimiter = ";";

    String score1 = "ds.e.pong.score1()";
    String score2 = "ds.e.pong.score2()";
    String numberCollisionsWithBall1 = "ds.e.pong.numberOfCollisionsWithBall1()";
    String numberCollisionsWithBall2 = "ds.e.pong.numberOfCollisionsWithBall2()";

    Map<String, Integer> opponentIndices = new HashMap<>();
    List<String> opponentNames = new ArrayList<>();
    List<Pair<String, NumericalDynamicalSystem<?>>> opponents = new ArrayList<>();
    int maxSeed = 0;

    List<String> lines1 = Files.readAllLines(Paths.get(CSVPath1));
    List<String> lines2 = new ArrayList<>();
    if (!singleCSV) {
      lines2 = Files.readAllLines(Paths.get(CSVPath2));
    }

    if (!singleCSV) {
      System.out.println("Removing header, check if correct: " + lines2.getFirst());
      lines2.removeFirst();
    }

    List<String> mergedLines = new ArrayList<>(lines1);
    if (!singleCSV) {
      mergedLines.addAll(lines2);
    }

    String mergedCSVContent = String.join(System.lineSeparator(), mergedLines);

    try (Reader mergedReader = new StringReader(mergedCSVContent)) {
      CSVParser csvParser = CSVFormat.Builder.create()
          .setDelimiter(delimiter)
          .build()
          .parse(mergedReader);

      List<CSVRecord> records = csvParser.getRecords();
      CSVRecord headerCSV = records.getFirst();

      int mapperColumnIndex = 0, genotypeColumnIndex = 0, nameColumnIndex = 0, seedColumnIndex = 0;
      for (int i = 0; i < headerCSV.size(); i++) {
        String columnName = headerCSV.get(i);
        if (columnName.contains("mapper")) {
          mapperColumnIndex = i;
        } else if (columnName.contains("genotype")) {
          genotypeColumnIndex = i;
        } else if (columnName.contains("name")) {
          nameColumnIndex = i;
        } else if (columnName.contains("seed")) {
          seedColumnIndex = i;
        }
      }

      Function<String, Object> deserializer = (Function<String, Object>) BUILDER.build("f.fromBase64()");
      PongEnvironment environment = (PongEnvironment) BUILDER.build("ds.e.pong()");
      NumericalDynamicalSystem<?> exampleNDS = environment.exampleAgent();

      for (int i = 1; i < records.size(); i++) {
        CSVRecord record = records.get(i);
        NumericalDynamicalSystem<?> opponent = ((InvertibleMapper<Object, NumericalDynamicalSystem<?>>) BUILDER.build(
            record.get(mapperColumnIndex)
        ))
            .mapperFor(exampleNDS)
            .apply(deserializer.apply(record.get(genotypeColumnIndex)));
        String name = record.get(nameColumnIndex);
        maxSeed = Math.max(maxSeed, Integer.parseInt(record.get(seedColumnIndex)));
        if (!opponentIndices.containsKey(name)) {
          opponentIndices.put(name, opponentNames.size());
          opponentNames.add(name);
        }
        opponents.add(new Pair<>(name, opponent));
      }

      opponentIndices.put("simple", opponentNames.size());
      opponentNames.add("simple");
      for (int i = 0; i < maxSeed; i++) {
        opponents.add(new Pair<>("simple", new PongAgent()));
      }

      int numOpponents = opponentNames.size();
      String[][] scoreMatrix = new String[numOpponents][numOpponents];
      String[][] collisionsMatrix = new String[numOpponents][numOpponents];
      int[][] matchCounts = new int[numOpponents][numOpponents];
      double[][] totalScore1 = new double[numOpponents][numOpponents];
      double[][] totalScore2 = new double[numOpponents][numOpponents];
      double[][] totalCollisions1 = new double[numOpponents][numOpponents];
      double[][] totalCollisions2 = new double[numOpponents][numOpponents];

      Map<String, Integer> winsMap = new HashMap<>();
      for (String name : opponentNames) {
        winsMap.put(name, 0);
      }

      HomogeneousBiAgentTask<NumericalDynamicalSystem<?>, double[], double[], PongEnvironment.State> task = HomogeneousBiAgentTask
          .fromHomogenousBiEnvironment(() -> environment, s -> false, new DoubleRange(0, 60), 0.05);

      for (Pair<String, NumericalDynamicalSystem<?>> opponent1 : opponents) {
        for (Pair<String, NumericalDynamicalSystem<?>> opponent2 : opponents) {
          int index1 = opponentIndices.get(opponent1.first());
          int index2 = opponentIndices.get(opponent2.first());

          BiSimulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>> matchOutcome = task
              .simulate(new Pair<>(opponent1.second(), opponent2.second()));

          double fitness1 = ((FormattedNamedFunction<BiSimulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>, Double>) BUILDER
              .build(score1)).apply(matchOutcome);
          double fitness2 = ((FormattedNamedFunction<BiSimulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>, Double>) BUILDER
              .build(score2)).apply(matchOutcome);
          double collisions1 = ((FormattedNamedFunction<BiSimulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>, Double>) BUILDER
              .build(numberCollisionsWithBall1)).apply(matchOutcome);
          double collisions2 = ((FormattedNamedFunction<BiSimulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>, Double>) BUILDER
              .build(numberCollisionsWithBall2)).apply(matchOutcome);

          totalScore1[index1][index2] += fitness1;
          totalScore2[index1][index2] += fitness2;
          totalCollisions1[index1][index2] += collisions1;
          totalCollisions2[index1][index2] += collisions2;
          matchCounts[index1][index2]++;

          if (fitness1 > fitness2) {
            winsMap.put(opponent1.first(), winsMap.get(opponent1.first()) + 1);
          } else if (fitness2 > fitness1) {
            winsMap.put(opponent2.first(), winsMap.get(opponent2.first()) + 1);
          }
        }
      }

      for (int i = 0; i < numOpponents; i++) {
        for (int j = 0; j < numOpponents; j++) {
          if (matchCounts[i][j] > 0) {
            scoreMatrix[i][j] = (" = " + (totalScore1[i][j] / matchCounts[i][j]) + " - " + (totalScore2[i][j] / matchCounts[i][j]))
                .replace(".", ",");
            collisionsMatrix[i][j] = (" = " + (totalCollisions1[i][j] / matchCounts[i][j]) + " - " + (totalCollisions2[i][j] / matchCounts[i][j]))
                .replace(".", ",");
          }
        }
      }

      List<Map.Entry<String, Integer>> ranking = new ArrayList<>(winsMap.entrySet());
      ranking.sort((e1, e2) -> Integer.compare(e2.getValue(), e1.getValue()));

      try (BufferedWriter rankingWriter = Files.newBufferedWriter(Paths.get(folder + "ranking.csv"))) {
        rankingWriter.write("nome;vittorie\n");
        for (Map.Entry<String, Integer> entry : ranking) {
          rankingWriter.write(entry.getKey() + ";" + entry.getValue() + "\n");
        }
      }

      try (BufferedWriter scoreWriter = Files.newBufferedWriter(
          Paths.get(folder + "scores-formula.csv")
      ); BufferedWriter collisionsWriter = Files.newBufferedWriter(Paths.get(folder + "collisions-formula.csv"))) {
        scoreWriter.write(";" + String.join(";", opponentNames) + "\n");
        collisionsWriter.write(";" + String.join(";", opponentNames) + "\n");
        for (int i = 0; i < numOpponents; i++) {
          scoreWriter.write(opponentNames.get(i) + ";" + String.join(";", scoreMatrix[i]) + "\n");
          collisionsWriter.write(opponentNames.get(i) + ";" + String.join(";", collisionsMatrix[i]) + "\n");
        }
      }
    }
  }
}
