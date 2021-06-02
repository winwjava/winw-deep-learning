package winw.ai.util.clustering;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;

/**
 * 将类似对象组成的多个类的过程被称为聚类。
 * <p>
 * 由聚类所生成的簇是一组数据对象的集合，这些对象与同一个簇中的对象彼此相似，与其他簇中的对象相异。
 * <p>
 * 聚类算法有：
 * <ol>
 * <li>划分方法(partitioning methods)：K-MEANS算法（K均值聚类）
 * <li>层次方法(hierarchical methods)：BIRCH算法、CURE算法、CHAMELEON算法
 * <li>基于密度的方法(density-based methods)：DBSCAN
 * </ol>
 * 
 * @author winw
 *
 */
public class ClustererTest {

	// 自然语言中大约有10万级别的类别。
	// 而类与类之间还存在着关系，这个关系，怎么表示？需要学习系统参与
	// 抽象是什么功能？

	// 概念系统，与聚类的关系？
	// 自然语言的处理：大脑额叶，运动皮层都有参与。
	
	public static void main(String[] args) {
		// ori is sample as n instances with m features, here n=8,m=2
		int ori[][] = { { 2, 5 }, { 6, 4 }, { 5, 3 }, { 2, 2 }, { 1, 4 }, { 5, 2 }, { 3, 3 }, { 2, 3 } };
		int n = 8;

		List<DoublePoint> points = new ArrayList<DoublePoint>();
		for (int i = 0; i < n; i++) {
			points.add(new DoublePoint(ori[i]));
		}
		KMeansPlusPlusClusterer<DoublePoint> kMeans = new KMeansPlusPlusClusterer<DoublePoint>(3);
		List<CentroidCluster<DoublePoint>> cluster = kMeans.cluster(points);
		for (CentroidCluster<DoublePoint> centroidCluster : cluster) {
			double[] point = centroidCluster.getCenter().getPoint();
			System.out.println(point[0] + ", " + point[1]);
		}
	}

}
