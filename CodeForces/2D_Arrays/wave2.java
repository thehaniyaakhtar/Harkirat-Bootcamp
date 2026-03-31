import java.util.*;
import java.io.*;

public class Main{
    public static void main(String[] args) throws Exception{
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        StringTokenizer st = new StringTokenizer(br.readLine());
        
        int n = Integer.parseInt(st.nextToken());
        int m = Integer.parseInt(st.nextToken());
        
        int[][] a = new int[n][m];
        
        for(int i = 0; i < n; i++){
            st = new StringTokenizer(br.readLine());
            for(int j = 0; j < m; j++){
                a[i][j] = Integer.parseInt(st.nextToken());
            }
        }
        
        StringBuilder sb = new StringBuilder();
        
        for(int j = 0; j < m; j++){
            if(j % 2 == 0){
                for(int i = 0; i < n; i++){
                    sb.append(a[i][j]).append(" ");
                }
            }
            else{
                for(int i = n-1; i >= 0; i--){
                    sb.append(a[i][j]).append(" ");
                }
            }
        }
        System.out.println(sb.toString());
    }
}
