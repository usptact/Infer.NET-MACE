using System;
using System.Collections.Generic;
using System.IO;

namespace MACE
{
    public class CsvReader
    {
        private StreamReader sr;
        private List<int[]> dataList;
        private int numWorkers;
        private int numItems;
        private int numCategories;

        public CsvReader(string fileName)
        {
            sr = new StreamReader(fileName);
            dataList = null;
            numWorkers = 0;
            numItems = 0;
            numCategories = 0;
        }

        //
        // Read in the CSV and populate data structures
        //
        public void read()
        {
            // get number of workers from the header
            string[] header = sr.ReadLine().Split(',');
            numWorkers = header.Length;

            dataList = new List<int[]>();
            while (!sr.EndOfStream)
            {
                string[] line = sr.ReadLine().Split(',');
                int[] temp = new int[numWorkers];
                for (int i = 0; i < line.Length; i++)
                {
                    if (line[i] == "")          // missing data is marked with "-1"
                        temp[i] = -1;
                    else
                    {
                        int value = Convert.ToInt32(line[i]);
                        numCategories = Math.Max(numCategories, value);
                        temp[i] = value;
                    }
                }
                dataList.Add(temp);
                numItems++;
            }
        }

        //
        // Return data: converts List<int[]> to int[][]
        //
        public int[][] getData()
        {
            if (dataList == null)
                return null;
            int[][] data = new int[numItems][];
            for (int i = 0; i < numItems; i++)
            {
                data[i] = new int[numWorkers];
                int[] item = dataList[i];
                for (int j = 0; j < numWorkers; j++)
                    data[i][j] = item[j];
            }
            return data;
        }

        public int getNumWorkers() { return numWorkers; }
        public int getNumItems() { return numItems; }
        public int getNumCategories() { return numCategories; }

    }
}
