package io.github.ericmedvet.jgea.core.solver.mapelites;

import io.github.ericmedvet.jgea.core.order.PartialComparator;
import io.github.ericmedvet.jgea.core.order.PartiallyOrderedCollection;
import io.github.ericmedvet.jnb.datastructure.Pair;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class M1Strategy implements CoMEStrategy {


  private List<Observation<Object>> bestObservations;

  public M1Strategy() {
  }

  @Override
  public List<Double> getOtherCoords(List<Double> theseCoords) {
      Observation<Object> theseCoordsObservation = bestObservations.stream().
          filter(obs -> obs.theseCoords().equals(theseCoords)).
          findFirst().
          orElse(null);
    assert theseCoordsObservation != null;
    if (theseCoordsObservation.otherCoords() == null) {
        return Collections.nCopies(theseCoords.size(), 0.5d);
      }
    return theseCoordsObservation.otherCoords();
  }

  @Override
  public <Q> void update(Collection<Observation<Q>> newObservations, PartialComparator<Q> qComparator) {

      Set<List<Double>> coordsToUpdate = newObservations.stream()
        .map(Observation::theseCoords)
        .collect(Collectors.toSet());

      for (List<Double> coords : coordsToUpdate) {
          Optional<Observation<Object>> bestObservationForCoords = bestObservations.stream()
              .filter(obs -> obs.theseCoords().equals(coords))
              .findFirst();

          List<Observation<Q>> newObservationsForCoords = (List<Observation<Q>>) newObservations.stream()
            .filter(obs -> obs.theseCoords().equals(coords)).toList();

          Optional<Observation<Q>> bestNewObservationForCoords =
            PartiallyOrderedCollection.from(newObservationsForCoords, qComparator.comparing(Observation::q))
                .firsts()
                .stream()
                .findAny();

          // if bestNewObservationForCoords is better then bestObservationForCoords
            // bestObservations.remove(obs s.t. theseCoords = coords)
            // bestObservations.add(bestNewObservationForCoords)
      }

      /*
    for (Observation<Q> singleObservation : newObservations){
      // List of Observations which theseCoords = singleObservation
      List<Observation<Object>> soObservations = bestObservations.stream().
          filter(obs -> obs.theseCoords().equals(singleObservation.theseCoords()))
          .toList();

      Optional<Observation<Q>> oBestObservation = PartiallyOrderedCollection.from(soObservations, qComparator.comparing(Observation.q()))
          .firsts()
          .stream()
          .findAny();

      if (oBestObservation.isPresent()) {
        Observation<Q> bestObservation = oBestObservation.get();
        if ((singleObservation.q() == null)
            || (qComparator.compare((Q) singleObservation.q(), bestObservation.q())
            == PartialComparator.PartialComparatorOutcome.BEFORE)) {
          bestObservations.removeIf(obs -> obs.theseCoords().equals(bestObservation.theseCoords()));
          bestObservations.add((Observation<Object>) bestObservation);
        }
      }

  }
    */

} // end of the update method



} // end of the class

