import java.util.*;
import java.io.*;

public class Main{
    public static void main(String[] args) throws Exception{
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        StringTokenizer st = new StringTokenizer(br.readLine());
        
        int n = Integer.parseInt(st.nextToken());
        int m = Integer.parseInt(st.nextToken());
        
        int[] colSum = new int[m];
        
        for(int i = 0; i < n; i++){
            st = new StringTokenizer(br.readLine());
            
            for(int j = 0; j < m; j++){
                int val = Integer.parseInt(st.nextToken());
                colSum[j] += val;
            }
        }

        StringBuilder result = new StringBuilder();
        for(int j = 0; j < m; j++){
            result.append(colSum[j]).append(" ");
        }
        System.out.println(result.toString().trim());
    }
}
