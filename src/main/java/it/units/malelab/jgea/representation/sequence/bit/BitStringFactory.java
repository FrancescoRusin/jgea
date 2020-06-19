/*
 * Copyright (C) 2020 Eric Medvet <eric.medvet@gmail.com> (as eric)
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  See the GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package it.units.malelab.jgea.representation.sequence.bit;

import it.units.malelab.jgea.core.IndependentFactory;

import java.util.Random;

/**
 * @author eric
 */
public class BitStringFactory implements IndependentFactory<BitString> {

  private final int size;

  public BitStringFactory(int size) {
    this.size = size;
  }

  @Override
  public BitString build(Random random) {
    BitString bitString = new BitString(size);
    for (int i = 0; i < size; i++) {
      bitString.set(i, random.nextBoolean());
    }
    return bitString;
  }

}