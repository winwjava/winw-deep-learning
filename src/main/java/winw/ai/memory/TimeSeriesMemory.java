package winw.ai.memory;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * 由概念组成。记录了概念的时序关系。
 * 
 * <p>
 * 记忆存放在N维网络中。
 * 
 * @author winw
 *
 */
public class TimeSeriesMemory {

	
	public static void main(String[] args) {
		INDArray tens = Nd4j.zeros(3,5).addi(10);
	}
}
