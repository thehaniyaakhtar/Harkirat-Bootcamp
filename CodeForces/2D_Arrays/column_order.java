import java.io.*;
import java.util.*;

public class Main{
    public static void main(String[] args) throws Exception{
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        StringTokenizer st;
        
        int n, m;
        st = new StringTokenizer(br.readLine());
        n = Integer.parseInt(st.nextToken());
        m = Integer.parseInt(st.nextToken());
        
        int[][] arr = new int[n][m];
        
        for(int i = 0; i < n; i++){
            st = new StringTokenizer(br.readLine());
            for(int j = 0; j < m; j++){
                arr[i][j] = Integer.parseInt(st.nextToken());
            }
        }
        
        StringBuilder sb = new StringBuilder();
        
        for(int j = 0; j < m; j++){
            for(int i = 0; i < n; i++){
                sb.append(arr[i][j]).append(" ");
            }
        }
        System.out.print(sb);
    }
}
