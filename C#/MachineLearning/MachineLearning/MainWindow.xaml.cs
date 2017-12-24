using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Accord;
using Accord.DataSets;
using Accord.Math.Geometry;
using Accord.Statistics.Filters;
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.DecisionTrees.Learning;
using Accord.MachineLearning.DecisionTrees.Rules;
using Accord.Math;
using Accord.Math.Optimization.Losses;

namespace MachineLearning
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();

            //Random Seed (to stay the same)
            Accord.Math.Random.Generator.Seed = 0;
        }

        private void button_Click(object sender, RoutedEventArgs e)
        {
            //Decided against the Iris set which is in Accord
            //var iris = new Iris();
            //double[][] inputs = iris.Instances;
            //int[] outputs = iris.ClassLabels;

            string[][] data = DataSet.CustomIris.iris_values.Split(new[] { "\r\n" },
                StringSplitOptions.RemoveEmptyEntries).Apply(x => x.Split(','));

            //features
            double[][] inputs = data.GetColumns(0, 1, 2, 3).To<double[][]>();

            //labels
            string[] labels = data.GetColumn(4);

            //Codebook translates any input into usable (integers) for the tree
            //var codebook = new Codification(outputs, inputs);
            var cb = new Codification("Output", labels);

            int[] outputs = cb.Transform("Output", labels);

            DecisionVariable[] features =
            {
                new DecisionVariable("sepal length", DecisionVariableKind.Continuous),
                new DecisionVariable("sepal width", DecisionVariableKind.Continuous),
                new DecisionVariable("petal length", DecisionVariableKind.Continuous),
                new DecisionVariable("petal width", DecisionVariableKind.Continuous),
            };

            var decisionTree = new DecisionTree(inputs: features, classes: 3);
            var c45learner = new C45Learning(decisionTree);
            c45learner.Learn(inputs, outputs);

            int[] estimated = decisionTree.Decide(inputs);            

            double error = new ZeroOneLoss(outputs).Loss(decisionTree.Decide(inputs));

            //Why rules?
            DecisionSet decisionSet = decisionTree.ToRules();
            
            string ruleText = decisionSet.ToString(cb, "Output",
                System.Globalization.CultureInfo.InvariantCulture);

            //var tree = new DecisionTree(inputs: features, classes: 3);

            #region UI
            //Set ouput to UI
            tb_output.Text = ruleText;

            //Calculate the flowers and input to UI -> TODO Bindings
            var setosaCount = 0;
            var versicolorCount = 0;
            var virginicaCount = 0;

            for (int i = 0; i < estimated.Length; i++)
            {
                if (estimated[i] == 0)
                {
                    setosaCount++;
                }
                if (estimated[i] == 1)
                {
                    versicolorCount++;
                }
                if (estimated[i] == 2)
                {
                    virginicaCount++;
                }
            }

            tb_setosa.Text = setosaCount.ToString();
            tb_versi.Text = versicolorCount.ToString();
            tb_virgi.Text = virginicaCount.ToString();
            #endregion UI
        }
    }
}
