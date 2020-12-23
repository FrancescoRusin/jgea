/*
 * Copyright 2020 Eric Medvet <eric.medvet@gmail.com> (as eric)
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

package it.units.malelab.jgea.core.evolver.speciation;

import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.order.PartiallyOrderedCollection;
import it.units.malelab.jgea.distance.Distance;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.Clusterable;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.distance.DistanceMeasure;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author federico
 */
public class KMeansSpeciator<G, S, F> implements Speciator<Individual<G, S, F>> {

  private KMeansPlusPlusClusterer<ClusterableIndividual> kMeans = null;
  private final int k;
  private final int maxIterations;
  private final Function<Individual<G, S, F>, double[]> converter;
  private final Distance<double[]> distance;

  private class ClusterableIndividual implements Clusterable {
    private final Individual<G, S, F> individual;
    private double[] point;

    public ClusterableIndividual(Individual<G, S, F> individual) {
      this.individual = individual;
    }

    @Override
    public double[] getPoint() {
      if (point == null) {
        point = converter.apply(individual);
      }
      return point;
    }

    public void setPoint(double[] newPoint) {
      System.arraycopy(newPoint, 0, point, 0, newPoint.length);
    }
  }

  public KMeansSpeciator(int k, int maxIterations, Distance<double[]> distance, Function<Individual<G, S, F>, double[]> converter) {
    if (k != -1) {
      this.kMeans = new KMeansPlusPlusClusterer<>(
              2,
              maxIterations,
              (DistanceMeasure) distance::apply
      );
    }
    this.k = k;
    this.maxIterations = maxIterations;
    this.converter = converter;
    this.distance = distance;
  }

  @Override
  public Collection<Species<Individual<G, S, F>>> speciate(PartiallyOrderedCollection<Individual<G, S, F>> population) {
    Collection<ClusterableIndividual> points = population.all().stream()
        .map(ClusterableIndividual::new)
        .collect(Collectors.toList());

    if (points.stream().mapToInt(p -> p.getPoint().length).distinct().count() != 1) {
      throw new RuntimeException("all points to be clustered must have same length");
    }

    normalizePoints(points);

    List<CentroidCluster<ClusterableIndividual>> clusters = clusterPoints(points);

    List<ClusterableIndividual> representers = clusters.stream().map(c -> {
      ClusterableIndividual closest = c.getPoints().get(0);
      double closestD = distance.apply(closest.point, c.getCenter().getPoint());
      for (int i = 0; i < c.getPoints().size(); i++) {
        double d = distance.apply(c.getPoints().get(i).point, c.getCenter().getPoint());
        if (d < closestD) {
          closestD = d;
          closest = c.getPoints().get(i);
        }
      }
      return closest;
    }).collect(Collectors.toList());
    return IntStream.range(0, clusters.size())
        .mapToObj(i -> new Species<>(
            clusters.get(i).getPoints().stream()
                .map(ci -> ci.individual)
                .collect(Collectors.toList()),
            representers.get(i).individual
        )).collect(Collectors.toList());
  }

  private void normalizePoints(Collection<ClusterableIndividual> points) {
    int length = points.stream().mapToInt(p -> p.getPoint().length).distinct().findAny().getAsInt();
    double[] maxValues = new double[length];
    double[] minValues = new double[length];
    for (int i=0; i < length; ++i) {
      int finalI = i;
      maxValues[i] = points.stream().mapToDouble(p -> p.getPoint()[finalI]).max().getAsDouble();
      minValues[i] = points.stream().mapToDouble(p -> p.getPoint()[finalI]).min().getAsDouble();
    }
    points.forEach(p -> {double[] oldPoint = p.getPoint();
      double[] temp = new double[length];
      for (int i=0; i < length; ++i) {
        if (maxValues[i] - minValues[i] == 0.0) {
          temp[i] = 1.0;
        }
        else {
          temp[i] = (oldPoint[i] - minValues[i]) / (maxValues[i] - minValues[i]);
        }
      }
      p.setPoint(temp);});
  }

  private List<CentroidCluster<ClusterableIndividual>> clusterPoints(Collection<ClusterableIndividual> points) {
    if (k != -1) {
      return kMeans.cluster(points);
    }
    double maxSilhouette = Double.NEGATIVE_INFINITY;
    List<CentroidCluster<ClusterableIndividual>> clusters = new ArrayList<>();
    for (int nClusters=2; nClusters < 10; ++nClusters) {
      List<CentroidCluster<ClusterableIndividual>> result = (new KMeansPlusPlusClusterer<ClusterableIndividual>(nClusters, maxIterations, (DistanceMeasure) distance::apply)).cluster(points);
      double silhouette = computeSilhouette(result, points.size());
      if (silhouette > maxSilhouette) {
        maxSilhouette = silhouette;
        clusters = result;
      }
    }
    return clusters;
  }

  private double computeSilhouette(List<CentroidCluster<ClusterableIndividual>> clusters, int n) {
    double[] s = new double[n];
    Map<double[], List<double[]>> points = clusters.stream().collect(Collectors.toMap(c -> c.getCenter().getPoint(),
            c -> c.getPoints().stream().map(Clusterable::getPoint).collect(Collectors.toList())));
    int k = 0;
    double a, b;
    for (Map.Entry<double[], List<double[]>> entry : points.entrySet()) {
      List<double[]> cluster = entry.getValue();
      for (double[] point : cluster) {
        if (cluster.size() == 1) {
          s[k++] = 0.0;
          continue;
        }
        a = cluster.stream().filter(p -> System.identityHashCode(p) != System.identityHashCode(point)).mapToDouble(p -> distance.apply(p, point)).average().getAsDouble();
        b = points.entrySet().stream().filter(e -> System.identityHashCode(e.getKey()) != System.identityHashCode(entry.getKey())).mapToDouble(e -> e.getValue().stream().mapToDouble(p -> distance.apply(p, point)).average().getAsDouble()).min().getAsDouble();
        if (a < b) {
          s[k++] = 1.0 - (a / b);
        }
        else if (a == b) {
          s[k++] = 0.0;
        }
        else {
          s[k++] = (b / a) - 1.0;
        }
      }
    }
    return Arrays.stream(s).average().getAsDouble();
  }

}
