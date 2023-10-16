
package io.github.ericmedvet.jgea.problem.extraction;

import com.google.common.collect.Range;
import io.github.ericmedvet.jgea.core.distance.Distance;

import java.util.Set;
public class ExtractionSetDistance implements Distance<Set<Range<Integer>>> {

  private final int length;
  private final int bins;

  public ExtractionSetDistance(int length, int bins) {
    this.length = length;
    this.bins = bins;
  }

  @Override
  public Double apply(Set<Range<Integer>> ranges1, Set<Range<Integer>> ranges2) {
    boolean[] mask1 = new boolean[bins + 1];
    boolean[] mask2 = new boolean[bins + 1];
    for (Range<Integer> range : ranges1) {
      mask1[(int) Math.floor((double) range.lowerEndpoint() / (double) length * (double) bins)] = true;
      mask1[(int) Math.floor((double) range.upperEndpoint() / (double) length * (double) bins)] = true;
    }
    for (Range<Integer> range : ranges2) {
      mask2[(int) Math.floor((double) range.lowerEndpoint() / (double) length * (double) bins)] = true;
      mask2[(int) Math.floor((double) range.upperEndpoint() / (double) length * (double) bins)] = true;
    }
    double count = 0;
    for (int i = 0; i < bins; i++) {
      count = count + ((mask1[i] == mask2[i]) ? 1 : 0);
    }
    return ((double) length - count) / (double) length;
  }

}
