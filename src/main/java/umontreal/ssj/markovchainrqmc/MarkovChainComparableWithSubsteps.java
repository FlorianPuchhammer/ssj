package umontreal.ssj.markovchainrqmc;

import umontreal.ssj.rng.RandomStream;
import umontreal.ssj.stat.Tally;
import umontreal.ssj.util.sort.MultiDimComparable;

public abstract class MarkovChainComparableWithSubsteps extends MarkovChainComparable
		implements MultiDimComparable<MarkovChainComparable> {
	public int numSubsteps;
	
	public void nextStep(RandomStream stream) {
		System.out.println("CAUTION: nextStep(stream) does nothing!!!");
		return;
	}

	public abstract void nextStep(RandomStream stream, int substep);

	@Override
	public void simulSteps (int numSteps, RandomStream stream) {
        initialState ();
        this.numSteps = numSteps;
        int step = 0;
        while (step < numSteps && !hasStopped()){
        	for(int substep = 0; substep < numSubsteps; substep++)
            nextStep (stream,substep);
            ++step;
        }
    }
}
